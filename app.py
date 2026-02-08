import json
import re
import time
from typing import List, Dict, Any, Tuple

import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI


# ===============================
# 유틸
# ===============================
def strip_b_tags(text: str) -> str:
    """네이버 지역검색 결과 title에 섞여오는 <b> 태그 제거"""
    if not text:
        return ""
    return re.sub(r"</?b>", "", text)


def get_secret_or_input(key: str, label: str, help_text: str = "", is_password: bool = True) -> str:
    """
    1) Streamlit Cloud에서는 st.secrets를 우선 사용
    2) 없으면 sidebar 입력으로 fallback
    """
    # st.secrets는 존재하지만 키가 없을 수도 있음
    if hasattr(st, "secrets") and key in st.secrets:
        return str(st.secrets[key])

    return st.sidebar.text_input(
        label,
        type="password" if is_password else "default",
        help=help_text,
    )


def naver_local_search(
    query: str,
    client_id: str,
    client_secret: str,
    display: int = 5,
    sort: str = "comment",
) -> List[Dict[str, str]]:
    """
    네이버 지역검색 API로 '실존' 장소 후보를 가져온다.
    문서: https://developers.naver.com/docs/serviceapi/search/local/local.md
    """
    url = "https://openapi.naver.com/v1/search/local.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    params = {
        "query": query,
        "display": max(1, min(display, 5)),  # 문서 기준 display 최대가 작은 편이라 안전하게
        "start": 1,
        "sort": sort,  # comment | random 등
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
    """이름+주소 기준으로 중복 제거"""
    seen = set()
    uniq = []
    for c in candidates:
        key = (c.get("name", "").strip(), c.get("address", "").strip())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="LunchMate 🍱", layout="wide")
st.title("🍽️ LunchMate")
st.caption("직장인의 상황과 선호도를 분석해 실제 존재하는 식당 후보 중 최적의 3곳을 추천합니다")

st.sidebar.header("🔐 API 설정")

# ✅ 배포 기준: Secrets에 넣는 걸 추천.
# - NAVER_CLIENT_ID
# - NAVER_CLIENT_SECRET
# - OPENAI_API_KEY (선택: 입력으로도 가능)
naver_client_id = get_secret_or_input(
    "NAVER_CLIENT_ID",
    "Naver Client ID",
    help_text="Streamlit Cloud라면 Secrets에 NAVER_CLIENT_ID로 저장해 두는 것을 권장합니다.",
)
naver_client_secret = get_secret_or_input(
    "NAVER_CLIENT_SECRET",
    "Naver Client Secret",
    help_text="Streamlit Cloud라면 Secrets에 NAVER_CLIENT_SECRET로 저장해 두는 것을 권장합니다.",
)
openai_api_key = get_secret_or_input(
    "OPENAI_API_KEY",
    "OpenAI API Key",
    help_text="Streamlit Cloud라면 Secrets에 OPENAI_API_KEY로 저장해 두는 것을 권장합니다.",
)

st.sidebar.header("🔍 검색 조건")
people = st.sidebar.slider("인원 수", 1, 10, 5)
distance = st.sidebar.selectbox("이동 거리", ["5분 이내", "10분 이내", "상관없음"])
food_type = st.sidebar.multiselect(
    "음식 종류",
    ["한식", "중식", "일식", "양식", "분식", "기타"],
    default=["한식"],
)

st.sidebar.caption(
    "⚠️ 네이버 지역검색 API만으로는 도보 5/10분을 정확히 계산하기 어려울 수 있어요.\n"
    "정확한 이동시간 필터링은 추후 Ncloud Maps Directions 연동을 추천합니다."
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
        situation = "팀원들과 조용히 대화할 수 있는 점심 회식 장소가 필요해요"
with col3:
    if st.button("🥣 해장 필요"):
        situation = "어제 술을 마셔서 해장에 좋은 음식을 먹고 싶어요"

st.write("")

# ===============================
# 추천 버튼 클릭
# ===============================
if st.button("🤖 점심 추천 받기"):
    if not situation:
        st.warning("상황을 입력해 주세요.")
        st.stop()

    if not (naver_client_id and naver_client_secret):
        st.warning("사이드바에 네이버 Client ID / Secret을 입력(또는 Secrets 설정)해 주세요.")
        st.stop()

    if not openai_api_key:
        st.warning("사이드바에 OpenAI API Key를 입력(또는 Secrets 설정)해 주세요.")
        st.stop()

    client = OpenAI(api_key=openai_api_key)

    # -------------------------------
    # 1) OpenAI로 검색 키워드(쿼리) 추출
    # -------------------------------
    query_schema = {
        "name": "LunchQueries",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "queries": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 6,
                    "items": {"type": "string"},
                }
            },
            "required": ["queries"],
        },
    }

    system_query_prompt = (
        "너는 네이버 지역검색 API에 넣을 '검색 키워드'를 만드는 도우미다.\n"
        "- 절대 식당 이름을 만들어내지 마라.\n"
        "- 짧고 검색에 잘 걸릴 키워드로만 2~6개를 제안하라.\n"
        "- 출력은 JSON만."
    )

    user_query_prompt = (
        f"상황: {situation}\n"
        f"인원: {people}\n"
        f"이동거리 선호: {distance}\n"
        f"선호 음식: {', '.join(food_type) if food_type else '상관없음'}\n\n"
        "네이버 지역검색에 넣을 검색어(queries) 2~6개를 만들어줘.\n"
        "예: '조용한 한식', '룸 있는 식당', '빠른 백반' 같은 형태."
    )

    with st.spinner("조건을 분석 중..."):
        q_resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system_query_prompt},
                {"role": "user", "content": user_query_prompt},
            ],
            response_format={"type": "json_schema", "json_schema": query_schema},
        )

    queries = json.loads(q_resp.output_text).get("queries", [])
    if not queries:
        st.error("검색 키워드 생성에 실패했어요. 잠시 후 다시 시도해 주세요.")
        st.stop()

    # -------------------------------
    # 2) 네이버 지역검색으로 '실존' 후보 수집
    # -------------------------------
    with st.spinner("주변 실제 식당 후보를 찾는 중..."):
        candidates: List[Dict[str, str]] = []
        for q in queries:
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
                time.sleep(0.1)  # 너무 공격적 호출 방지(가벼운 템포 조절)
            except requests.HTTPError as e:
                st.error(f"네이버 검색 API 호출 실패: {e}")
                st.stop()
            except requests.RequestException as e:
                st.error(f"네트워크 오류: {e}")
                st.stop()

        candidates = dedupe_candidates(candidates)

    if not candidates:
        st.warning("조건에 맞는 식당 후보를 찾지 못했어요. 키워드를 넓혀 다시 시도해 주세요.")
        st.stop()

    # -------------------------------
    # 3) OpenAI가 후보 중에서만 Top3 선택 + 이유 생성
    #    (중요: 후보 밖 식당 추천 금지)
    # -------------------------------
    rec_schema = {
        "name": "LunchRecommendations",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {"type": "string"},
                "recommendations": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "rank": {"type": "integer"},
                            "name": {"type": "string"},
                            "reason": {"type": "string"},
                            "address": {"type": "string"},
                            "category": {"type": "string"},
                            "tel": {"type": "string"},
                            "link": {"type": "string"},
                        },
                        "required": ["rank", "name", "reason", "address", "category", "tel", "link"],
                    },
                },
            },
            "required": ["summary", "recommendations"],
        },
    }

    # 후보를 너무 많이 주면 모델이 헷갈릴 수 있어 가까운 것 기준이 없으니 일단 상위 N개로 제한
    candidate_payload = candidates[:20]

    system_rec_prompt = (
        "너는 점심 추천 큐레이터다.\n"
        "반드시 제공된 candidates 목록에 있는 식당만 추천할 수 있다.\n"
        "candidates에 없는 식당 이름을 새로 만들거나 추천하면 실패다.\n"
        "사용자 상황과 조건에 맞춰 최대 3개를 고르고, 이유를 간결하게 설명하라.\n"
        "숫자(평점/가격/거리/시간)는 근거 데이터가 없으면 절대 지어내지 마라.\n"
        "출력은 JSON만."
    )

    user_rec_prompt = json.dumps(
        {
            "situation": situation,
            "people": people,
            "distance_pref": distance,
            "food_type": food_type,
            "candidates": candidate_payload,
        },
        ensure_ascii=False,
    )

    with st.spinner("후보 중에서 최적의 3곳을 고르는 중..."):
        r_resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system_rec_prompt},
                {"role": "user", "content": user_rec_prompt},
            ],
            response_format={"type": "json_schema", "json_schema": rec_schema},
        )

    result = json.loads(r_resp.output_text)
    recommendations = result.get("recommendations", [])
    summary = result.get("summary", "추천 결과를 확인해 주세요.")

    if not recommendations:
        st.error("추천 결과를 생성하지 못했어요. 다시 시도해 주세요.")
        st.stop()

    # rank 정렬 보정
    recommendations = sorted(recommendations, key=lambda x: x.get("rank", 999))

    # -------------------------------
    # 출력 UI
    # -------------------------------
    st.success(f"✅ **{summary}**")

    st.subheader("🏆 추천 식당 TOP (실존 후보 기반)")
    for r in recommendations:
        with st.container():
            st.markdown(f"### {r['rank']}️⃣ {r['name']}")
            st.write(f"📌 추천 이유: {r['reason']}")
            st.write(f"🏷️ 카테고리: {r.get('category','') or '정보 없음'}")
            st.write(f"📍 주소: {r.get('address','') or '정보 없음'}")
            st.write(f"☎️ 전화: {r.get('tel','') or '정보 없음'}")
            if r.get("link"):
                st.markdown(f"🔗 링크: {r['link']}")
            st.divider()

    # -------------------------------
    # 비교(간단 표 + 차트)
    # 네이버 API는 평점 수치를 제공하지 않는 경우가 많아서, '카테고리/전화 유무' 정도만 시각화
    # -------------------------------
    st.subheader("📊 후보 비교(기본 정보)")
    df = pd.DataFrame(recommendations)
    st.dataframe(
        df[["rank", "name", "category", "address", "tel"]],
        use_container_width=True,
        hide_index=True,
    )

    # 카테고리 길이(정보량) 같은 '임시 지표'를 차트로. (원치 않으면 제거 가능)
    st.subheader("📈 정보량 비교(카테고리 텍스트 길이)")
    df["category_len"] = df["category"].fillna("").apply(len)

    fig, ax = plt.subplots()
    ax.bar(df["name"], df["category_len"])
    ax.set_ylabel("카테고리 텍스트 길이")
    st.pyplot(fig)

else:
    st.info("👆 상황을 입력하고 **점심 추천 받기** 버튼을 눌러주세요.")
