import json
import re
import time
from typing import List, Dict

import requests
import streamlit as st
import pandas as pd
from openai import OpenAI


# ===============================
# 유틸
# ===============================
def strip_b_tags(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"</?b>", "", text)


def get_secret(key: str) -> str:
    """Streamlit Cloud Secrets에서만 읽기 (사이드바 입력 제거)"""
    return str(st.secrets.get(key, "")).strip()


def naver_local_search(
    query: str,
    client_id: str,
    client_secret: str,
    display: int = 5,
    sort: str = "comment",
) -> List[Dict[str, str]]:
    """
    네이버 지역검색 API로 실존 장소 후보 수집
    https://openapi.naver.com/v1/search/local.json
    """
    url = "https://openapi.naver.com/v1/search/local.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    params = {
        "query": query,
        "display": max(1, min(display, 5)),
        "start": 1,
        "sort": sort,
    }
    r = requests.get(url, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    results: List[Dict[str, str]] = []
    for it in data.get("items", []):
        results.append(
            {
                "name": strip_b_tags(it.get("title", "")),
                "address": it.get("roadAddress") or it.get("address") or "",
                "category": it.get("category", ""),
                "tel": it.get("telephone", ""),
                "link": it.get("link", ""),
            }
        )
    return results


def dedupe_candidates(candidates: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    uniq = []
    for c in candidates:
        key = (c.get("name", "").strip(), c.get("address", "").strip())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def extract_json_from_text(text: str) -> dict:
    """
    모델이 JSON 외 텍스트를 섞었을 때를 대비해,
    가장 바깥 JSON 객체를 찾아 파싱 시도.
    """
    text = text.strip()

    # 이미 JSON이면 바로
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 코드블록 제거
    text = text.replace("```json", "```").replace("```", "")

    # 첫 { 부터 마지막 } 까지 추출
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    candidate = text[start : end + 1]
    return json.loads(candidate)


def llm_json(client: OpenAI, system: str, user: str, model: str = "gpt-4.1-mini", retries: int = 2) -> dict:
    """
    chat.completions 기반 JSON 응답 강제.
    SDK 호환성을 위해 response_format(json_schema) 대신 프롬프트로 강제하고,
    파싱 실패 시 짧게 재시도.
    """
    for attempt in range(retries + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        text = resp.choices[0].message.content or ""
        try:
            return extract_json_from_text(text)
        except json.JSONDecodeError:
            if attempt == retries:
                raise
            # 재시도: 더 강하게 “JSON만” 요구
            user = (
                user
                + "\n\n너의 직전 출력은 JSON 파싱에 실패했어. "
                  "다른 텍스트 없이 JSON만 다시 출력해."
            )
    raise RuntimeError("Unreachable")


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="LunchMate 🍱", layout="wide")
st.title("🍽️ LunchMate")
st.caption("직장인의 상황과 선호도를 분석해 ‘실제로 존재하는’ 식당 후보 중 최적의 3곳을 추천합니다")

# Secrets 상태 표시(입력칸 없음)
st.sidebar.header("🔐 연결 상태")
naver_client_id = get_secret("NAVER_CLIENT_ID")
naver_client_secret = get_secret("NAVER_CLIENT_SECRET")
openai_api_key = get_secret("OPENAI_API_KEY")

st.sidebar.write("네이버 API:", "✅" if (naver_client_id and naver_client_secret) else "❌ (Secrets 필요)")
st.sidebar.write("OpenAI API:", "✅" if openai_api_key else "❌ (Secrets 필요)")
st.sidebar.caption("Streamlit Cloud → Settings → Secrets 에 키를 넣어야 합니다.")

st.sidebar.header("🔍 검색 조건")
people = st.sidebar.slider("인원 수", 1, 10, 5)
distance = st.sidebar.selectbox("이동 거리", ["5분 이내", "10분 이내", "상관없음"])
food_type = st.sidebar.multiselect(
    "음식 종류",
    ["한식", "중식", "일식", "양식", "분식", "기타"],
    default=["한식"],
)

st.subheader("📝 오늘의 상황을 입력해 주세요")
situation = st.text_area(
    "자연스럽게 입력해 주세요",
    placeholder="예: 오늘 팀장님 모시고 5명이서 조용히 1시간 내로 먹어야 해요",
)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("⚡ 빨리 먹기"):
        situation = "시간이 없어서 빨리 먹을 수 있는 곳을 찾고 있어요"
with col2:
    if st.button("👥 팀 회식"):
        situation = "팀장님/팀원들과 조용히 대화할 수 있는 점심 회식 장소가 필요해요"
with col3:
    if st.button("🥣 해장 필요"):
        situation = "어제 술을 마셔서 해장에 좋은 음식을 먹고 싶어요"

st.write("")

# ===============================
# 추천 버튼
# ===============================
if st.button("🤖 점심 추천 받기"):
    if not situation:
        st.warning("상황을 입력해 주세요.")
        st.stop()

    if not (naver_client_id and naver_client_secret):
        st.error("네이버 Client ID/Secret이 없습니다. Streamlit Cloud의 Secrets에 등록해 주세요.")
        st.stop()

    if not openai_api_key:
        st.error("OpenAI API Key가 없습니다. Streamlit Cloud의 Secrets에 OPENAI_API_KEY로 등록해 주세요.")
        st.stop()

    client = OpenAI(api_key=openai_api_key)

    # 1) OpenAI로 '네이버 지역검색에 넣을 검색어' 생성
    system_query = (
        "너는 네이버 지역검색 API에 넣을 '검색어'를 생성하는 도우미다.\n"
        "- 식당 이름을 절대 만들지 마라.\n"
        "- 검색에 잘 걸릴 짧은 키워드 조합만 만들어라.\n"
        "- 출력은 JSON만. 스키마:\n"
        "{ \"queries\": [\"...\", \"...\"] }\n"
        "- queries는 2~6개."
    )
    user_query = (
        f"상황: {situation}\n"
        f"인원: {people}\n"
        f"이동거리 선호: {distance}\n"
        f"선호 음식: {', '.join(food_type) if food_type else '상관없음'}\n\n"
        "네이버 지역검색에 넣을 queries 2~6개를 만들어줘."
    )

    with st.spinner("조건을 분석 중..."):
        try:
            q_data = llm_json(client, system_query, user_query)
            queries = q_data.get("queries", [])
        except Exception:
            st.error("검색어 생성에 실패했어요. (OpenAI 응답 파싱 실패)")
            st.stop()

    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    if not queries:
        st.warning("검색어를 만들지 못했어요. 입력을 조금 더 구체적으로 적어주세요.")
        st.stop()

    # 2) 네이버 지역검색으로 '실존 후보' 수집
    with st.spinner("주변 실제 식당 후보를 찾는 중..."):
        candidates: List[Dict[str, str]] = []
        for q in queries[:6]:
            try:
                candidates.extend(
                    naver_local_search(
                        query=q,
                        client_id=naver_client_id,
                        client_secret=naver_client_secret,
                        display=5,
                        sort="comment",
                    )
                )
                time.sleep(0.08)
            except requests.HTTPError as e:
                st.error(f"네이버 지역검색 API 호출 실패(HTTP): {e}")
                st.stop()
            except requests.RequestException as e:
                st.error(f"네트워크 오류: {e}")
                st.stop()

        candidates = dedupe_candidates(candidates)

    if not candidates:
        st.warning("조건에 맞는 실제 식당 후보를 찾지 못했어요. 키워드를 넓혀 다시 시도해 주세요.")
        st.stop()

    # 3) 후보 안에서만 TOP3 추천 + 이유 생성 (후보 밖 금지)
    system_rec = (
        "너는 점심 추천 큐레이터다.\n"
        "- 반드시 candidates 목록에 있는 식당만 추천할 수 있다.\n"
        "- candidates에 없는 식당을 새로 만들면 실패다.\n"
        "- 숫자(평점/가격/거리/시간)는 근거 데이터가 없으면 절대 지어내지 마라.\n"
        "- 출력은 JSON만. 스키마:\n"
        "{\n"
        "  \"summary\": \"한 줄 결론\",\n"
        "  \"recommendations\": [\n"
        "    {\"rank\": 1, \"name\": \"...\", \"reason\": \"...\", \"address\": \"...\", \"category\": \"...\", \"tel\": \"...\", \"link\": \"...\"}\n"
        "  ]\n"
        "}\n"
        "- recommendations는 1~3개, rank는 1부터."
    )

    payload = {
        "situation": situation,
        "people": people,
        "distance_pref": distance,
        "food_type": food_type,
        "candidates": candidates[:25],  # 너무 길면 혼란 -> 제한
    }
    user_rec = json.dumps(payload, ensure_ascii=False)

    with st.spinner("후보 중에서 최적의 3곳을 고르는 중..."):
        try:
            r_data = llm_json(client, system_rec, user_rec)
        except Exception:
            st.error("추천 결과 생성에 실패했어요. (OpenAI 응답 파싱 실패)")
            st.stop()

    summary = r_data.get("summary", "추천 결과를 확인해 주세요.")
    recommendations = r_data.get("recommendations", [])

    if not isinstance(recommendations, list) or len(recommendations) == 0:
        st.error("추천 결과 형식이 올바르지 않습니다. 다시 시도해 주세요.")
        st.stop()

    # 정렬 및 최대 3개로 강제
    recommendations = [r for r in recommendations if isinstance(r, dict)]
    recommendations = sorted(recommendations, key=lambda x: int(x.get("rank", 999)))
    recommendations = recommendations[:3]

    # UI 출력
    st.success(f"✅ **{summary}**")

    st.subheader("🏆 추천 식당 TOP 3 (네이버 후보 기반)")
    for r in recommendations:
        with st.container():
            st.markdown(f"### {r.get('rank', '')}️⃣ {r.get('name', '이름 없음')}")
            st.write(f"📌 추천 이유: {r.get('reason', '')}")
            st.write(f"🏷️ 카테고리: {r.get('category', '') or '정보 없음'}")
            st.write(f"📍 주소: {r.get('address', '') or '정보 없음'}")
            st.write(f"☎️ 전화: {r.get('tel', '') or '정보 없음'}")
            if r.get("link"):
                st.markdown(f"🔗 링크: {r['link']}")
            st.divider()

    st.subheader("📋 비교 표")
    df = pd.DataFrame(recommendations)
    cols = [c for c in ["rank", "name", "category", "address", "tel", "link"] if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)

    # 간단 차트(“카테고리 정보량”처럼 검증 가능한 값만)
    st.subheader("📈 정보량 비교(카테고리 글자수)")
    df["category_len"] = df.get("category", "").fillna("").astype(str).apply(len)
    st.bar_chart(df.set_index("name")["category_len"])

else:
    st.info("👆 상황을 입력하고 **점심 추천 받기** 버튼을 눌러주세요.")
