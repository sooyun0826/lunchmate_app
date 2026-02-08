import json
import re
import time
import html
from typing import List, Dict, Any
from urllib.parse import quote_plus

import requests
import streamlit as st
import pandas as pd
from openai import OpenAI


# ===============================
# 기본 디폴트 설정 (초기값)
# ===============================
DEFAULT_PEOPLE = 2
DISTANCE_OPTIONS = ["5분 이내", "10분 이내", "상관없음"]
DEFAULT_DISTANCE_INDEX = 2  # "상관없음"

# ✅ “음식점 + 카페”까지 포함하도록 옵션 확장
FOOD_OPTIONS = [
    "한식", "중식", "일식", "양식", "분식", "기타",
    "카페", "디저트"
]
DEFAULT_FOOD_TYPES: List[str] = []

# ✅ 추천 개수
TOP_K = 5

# 후보 확장 / 성능
MAX_QUERIES = 6
LOCAL_DISPLAY_PER_QUERY = 5
CANDIDATE_LIMIT_FOR_LLM = 40
REQUEST_SLEEP_SEC = 0.08


# ===============================
# 유틸
# ===============================
def strip_b_tags(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"</?b>", "", text)


def get_secret(key: str) -> str:
    return str(st.secrets.get(key, "")).strip()


def safe_int(x: Any, default: int = 999) -> int:
    try:
        return int(x)
    except Exception:
        return default


def naver_local_search(
    query: str,
    client_id: str,
    client_secret: str,
    display: int = 5,
    sort: str = "comment",
    start: int = 1,
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
        "start": max(1, start),
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


def filter_candidates(candidates: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    ✅ 음식점/카페 추천 서비스 기준으로, 명백히 비식음료 업종만 제거
    (병원/학원/부동산 등)
    """
    bad_keywords = [
        "학원", "공인중개", "부동산", "미용", "네일", "피부", "성형",
        "헬스", "요가", "필라테스", "세탁", "수리", "정비", "렌탈",
        "교회", "성당", "절", "약국", "병원", "의원", "치과", "한의원",
        "주유", "자동차", "인테리어", "가구", "마트",
    ]
    out = []
    for c in candidates:
        name = (c.get("name") or "").strip()
        cat = (c.get("category") or "").strip()
        blob = f"{name} {cat}"
        if any(k in blob for k in bad_keywords):
            continue
        out.append(c)
    return out


def extract_json_from_text(text: str) -> dict:
    text = (text or "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    text = text.replace("```json", "```").replace("```", "")

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    candidate = text[start : end + 1]
    return json.loads(candidate)


def llm_json(
    client: OpenAI,
    system: str,
    user: str,
    model: str = "gpt-4.1-mini",
    retries: int = 2
) -> dict:
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
            user = user + "\n\n다른 텍스트 없이 JSON만 다시 출력해."
    raise RuntimeError("Unreachable")


def ensure_k_recommendations(
    recommendations: List[Dict[str, Any]],
    candidates: List[Dict[str, str]],
    k: int,
) -> List[Dict[str, Any]]:
    """
    추천 결과가 k개 미만이면 candidates에서 부족분을 채워 k개로 맞춤
    - 중복(이름+주소) 제거
    - rank 1~k 재정렬
    """
    def _key(name: str, address: str) -> tuple:
        return (str(name or "").strip(), str(address or "").strip())

    recs = [r for r in recommendations if isinstance(r, dict)]
    recs = sorted(recs, key=lambda x: safe_int(x.get("rank", 999)))

    picked = set()
    cleaned = []
    for r in recs:
        k0 = _key(r.get("name", ""), r.get("address", ""))
        if k0 in picked:
            continue
        picked.add(k0)
        cleaned.append(r)
    recs = cleaned

    if len(recs) < k:
        for c in candidates:
            k0 = _key(c.get("name", ""), c.get("address", ""))
            if k0 in picked:
                continue
            picked.add(k0)
            recs.append({
                "rank": len(recs) + 1,
                "name": c.get("name", ""),
                "reason": "후보 중 조건과 무난하게 잘 맞는 선택지입니다.",
                "tags": ["무난", "후보기반"],
                "address": c.get("address", ""),
                "category": c.get("category", ""),
                "link": c.get("link", ""),
                "evidence": ["네이버 지역검색 후보에 존재"],
            })
            if len(recs) == k:
                break

    recs = recs[:k]
    for i, r in enumerate(recs, start=1):
        r["rank"] = i
    return recs


def make_review_query(name: str, address: str) -> str:
    name = (name or "").strip()
    address = (address or "").strip()
    addr_hint = " ".join(address.split()[:3])
    q = f"{name} {addr_hint} 후기".strip()
    return re.sub(r"\s+", " ", q)


@st.cache_data(ttl=3600, show_spinner=False)
def naver_blog_search_cached(
    query: str,
    client_id: str,
    client_secret: str,
    display: int = 3,
    sort: str = "sim",
):
    url = "https://openapi.naver.com/v1/search/blog.json"
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

    items = []
    for it in data.get("items", []):
        items.append({
            "title": strip_b_tags(html.unescape(it.get("title", ""))),
            "link": it.get("link", ""),
            "desc": strip_b_tags(html.unescape(it.get("description", ""))),
            "thumbnail": it.get("thumbnail", ""),
        })
    return items


def naver_map_search_url(place_name: str, address: str = "") -> str:
    q = (place_name or "").strip()
    if address:
        q = f"{q} {address.split()[0]}".strip()
    return f"https://map.naver.com/v5/search/{quote_plus(q)}"


def build_cache_key(payload: Dict[str, Any]) -> str:
    compact = {
        "start": payload.get("start_location", ""),
        "situation": payload.get("situation", ""),
        "people": payload.get("people", 0),
        "distance": payload.get("distance_pref", ""),
        "food": payload.get("food_type", []),
        "exclude": payload.get("exclude", ""),
        "prefer": payload.get("prefer", ""),
        "visit_type": payload.get("visit_type", ""),
    }
    return json.dumps(compact, ensure_ascii=False, sort_keys=True)


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="PlaceMate 🍽️☕", layout="wide")

# ✅ 문구/네이밍: 점심 한정 제거
st.title("🍽️ PlaceMate ☕")
st.caption("아침/점심/저녁 상관없이 음식점과 카페를 ‘실제 존재하는’ 후보 중에서 추천해 드립니다.")

naver_client_id = get_secret("NAVER_CLIENT_ID")
naver_client_secret = get_secret("NAVER_CLIENT_SECRET")
openai_api_key = get_secret("OPENAI_API_KEY")

if "candidate_cache_key" not in st.session_state:
    st.session_state["candidate_cache_key"] = None
if "candidates" not in st.session_state:
    st.session_state["candidates"] = []


# ===============================
# 사이드바
# ===============================
st.sidebar.header("🕒 방문 목적(아침/점심/저녁/카페)")
visit_type = st.sidebar.selectbox(
    "추천 받을 종류",
    ["상관없음", "아침", "점심", "저녁", "카페/디저트"],
    index=0
)

st.sidebar.header("📍 출발 위치(정확도 개선)")
start_location = st.sidebar.text_input("출발지(회사/역/주소)", placeholder="예: 신촌역, 강남역, 판교역")

st.sidebar.header("🔍 검색 조건")
people = st.sidebar.slider("인원 수", 1, 10, DEFAULT_PEOPLE)
distance = st.sidebar.selectbox("이동 거리", DISTANCE_OPTIONS, index=DEFAULT_DISTANCE_INDEX)
food_type = st.sidebar.multiselect("음식/카페 종류", FOOD_OPTIONS, default=DEFAULT_FOOD_TYPES)

st.sidebar.header("🚫 제외 / ✅ 선호")
exclude_text = st.sidebar.text_input("제외 조건(쉼표로 구분)", placeholder="예: 매운 음식, 회, 웨이팅")
prefer_text = st.sidebar.text_input("선호 조건(쉼표로 구분)", placeholder="예: 조용한 곳, 가성비, 디저트")

st.sidebar.header("🖼️ 후기/사진 설정")
show_reviews = st.sidebar.checkbox("블로그 후기 표시", value=True)
review_display = st.sidebar.slider("장소당 블로그 후기 개수", 1, 3, 2)
blog_sort = st.sidebar.radio("후기 정렬", ["연관도(추천)", "최신순"], index=0)
blog_sort_param = "sim" if blog_sort.startswith("연관도") else "date"


# ===============================
# 메인 입력
# ===============================
st.subheader("📝 희망 조건을 자유롭게 입력해 주세요")
situation = st.text_area(
    "자연스럽게 입력해 주세요(취향, 방문 지역, 방문자 수, 상황 등)",
    placeholder="예: 오늘 저녁에 강남역 근처에서 조용한 파스타집 추천해줘. / 오후에 디저트 카페 가고 싶어.",
)

# ✅ 빠른 입력 버튼도 “점심” 전용 표현 제거
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("⚡ 빨리 먹기"):
        situation = "시간이 없어서 빨리 이용할 수 있는 곳을 찾고 있어요"
with col2:
    if st.button("👥 모임/회식"):
        situation = "여럿이 조용히 대화할 수 있는 모임 장소가 필요해요"
with col3:
    if st.button("🥣 해장"):
        situation = "어제 술을 마셔서 해장에 좋은 음식을 먹고 싶어요"
with col4:
    if st.button("☕ 카페"):
        situation = "디저트/커피가 괜찮고 사진 찍기 좋은 카페를 찾고 있어요"

st.write("")


def require_secrets_or_stop():
    if not (naver_client_id and naver_client_secret and openai_api_key):
        st.error("서비스 설정 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")
        st.stop()


def generate_queries(client: OpenAI, payload: Dict[str, Any]) -> List[str]:
    system_query = (
        "너는 네이버 지역검색 API에 넣을 '검색어(queries)'를 생성하는 도우미다.\n"
        "- 장소 이름을 절대 만들어내지 마라.\n"
        "- 반드시 사용자의 출발지(지역/역/주소) 정보를 queries에 반영하라.\n"
        "- 추천 받을 종류(아침/점심/저녁/카페) 정보를 반영하라.\n"
        "- 검색에 잘 걸리는 짧은 키워드 조합으로.\n"
        "- 출력은 JSON만. 스키마: { \"queries\": [\"...\", \"...\"] }\n"
        "- queries는 3~6개."
    )

    food = payload.get("food_type") or []
    food_str = ", ".join(food) if food else "(선택 없음)"

    visit = payload.get("visit_type", "상관없음")
    visit_hint = {
        "아침": "아침/브런치",
        "점심": "점심 맛집",
        "저녁": "저녁/술자리",
        "카페/디저트": "카페/디저트",
        "상관없음": "맛집/카페",
    }.get(visit, "맛집/카페")

    user_query = (
        f"추천 종류: {visit} ({visit_hint})\n"
        f"출발지: {payload.get('start_location') or '(미입력)'}\n"
        f"상황: {payload.get('situation')}\n"
        f"인원: {payload.get('people')}\n"
        f"이동거리 선호: {payload.get('distance_pref')}\n"
        f"선호 음식/카페 종류: {food_str}\n"
        f"제외 조건: {payload.get('exclude') or '(없음)'}\n"
        f"선호 조건: {payload.get('prefer') or '(없음)'}\n\n"
        "네이버 지역검색에 넣을 queries 3~6개를 만들어줘."
    )

    q_data = llm_json(client, system_query, user_query)
    queries = q_data.get("queries", [])
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]

    start = (payload.get("start_location") or "").strip()
    if start:
        patched = []
        for q in queries:
            patched.append(q if start in q else f"{start} {q}".strip())
        queries = patched

    uniq = []
    seen = set()
    for q in queries:
        if q in seen:
            continue
        seen.add(q)
        uniq.append(q)
    return uniq[:MAX_QUERIES]


def collect_candidates(queries: List[str]) -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []
    for q in queries[:MAX_QUERIES]:
        candidates.extend(
            naver_local_search(
                query=q,
                client_id=naver_client_id,
                client_secret=naver_client_secret,
                display=LOCAL_DISPLAY_PER_QUERY,
                sort="comment",
                start=1,
            )
        )
        time.sleep(REQUEST_SLEEP_SEC)

    candidates = dedupe_candidates(candidates)
    candidates = filter_candidates(candidates)
    return candidates


def recommend_from_candidates(client: OpenAI, payload: Dict[str, Any], candidates: List[Dict[str, str]]) -> Dict[str, Any]:
    system_rec = (
        "너는 음식점/카페 추천 큐레이터다.\n"
        "- 반드시 candidates 목록에 있는 장소만 추천할 수 있다.\n"
        "- candidates에 없는 장소를 새로 만들면 실패다.\n"
        "- 숫자(평점/가격/거리/시간)는 근거 데이터가 없으면 절대 지어내지 마라.\n"
        "- 출력은 JSON만.\n"
        "스키마:\n"
        "{\n"
        "  \"summary\": \"한 줄 결론\",\n"
        "  \"recommendations\": [\n"
        "    {\n"
        "      \"rank\": 1,\n"
        "      \"name\": \"...\",\n"
        "      \"reason\": \"...\",\n"
        "      \"tags\": [\"#브런치\", \"#조용함\", \"#디저트\"],\n"
        "      \"evidence\": [\"candidates에 존재\", \"카테고리: ...\", \"주소: ...\"],\n"
        "      \"address\": \"...\",\n"
        "      \"category\": \"...\",\n"
        "      \"link\": \"...\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        f"- recommendations는 1~{TOP_K}개, rank는 1부터."
    )

    llm_payload = {
        "visit_type": payload.get("visit_type", "상관없음"),
        "start_location": payload.get("start_location", ""),
        "situation": payload.get("situation", ""),
        "people": payload.get("people", 0),
        "distance_pref": payload.get("distance_pref", ""),
        "food_type": payload.get("food_type", []),
        "exclude": payload.get("exclude", ""),
        "prefer": payload.get("prefer", ""),
        "candidates": candidates[:CANDIDATE_LIMIT_FOR_LLM],
        "top_k": TOP_K,
    }
    user_rec = json.dumps(llm_payload, ensure_ascii=False)
    return llm_json(client, system_rec, user_rec)


# ===============================
# 버튼 (UX: 재추천)
# ===============================
btn1, btn2 = st.columns([1, 1])
with btn1:
    run_search = st.button("🤖 추천 받기", use_container_width=True)
with btn2:
    reroll = st.button("🔄 후보 그대로 다시 추천", use_container_width=True)


# ===============================
# 실행
# ===============================
if run_search or reroll:
    if not situation:
        st.warning("상황을 입력해 주세요.")
        st.stop()

    require_secrets_or_stop()
    client = OpenAI(api_key=openai_api_key)

    payload = {
        "visit_type": visit_type,
        "start_location": start_location.strip(),
        "situation": situation.strip(),
        "people": people,
        "distance_pref": distance,
        "food_type": food_type,
        "exclude": exclude_text.strip(),
        "prefer": prefer_text.strip(),
    }
    cache_key = build_cache_key(payload)

    if reroll and st.session_state.get("candidates") and st.session_state.get("candidate_cache_key") == cache_key:
        candidates = st.session_state["candidates"]
    else:
        with st.spinner("조건을 분석 중..."):
            try:
                queries = generate_queries(client, payload)
            except Exception:
                st.error("검색어 생성에 실패했어요. (OpenAI 응답 파싱 실패)")
                st.stop()

        if not queries:
            st.warning("검색어를 만들지 못했어요. 입력을 조금 더 구체적으로 적어주세요.")
            st.stop()

        with st.spinner("주변 실제 후보(음식점/카페)를 찾는 중..."):
            try:
                candidates = collect_candidates(queries)
            except requests.HTTPError as e:
                st.error(f"네이버 지역검색 API 호출 실패(HTTP): {e}")
                st.stop()
            except requests.RequestException as e:
                st.error(f"네트워크 오류: {e}")
                st.stop()

        if not candidates:
            st.warning("조건에 맞는 실제 후보를 찾지 못했어요. 키워드를 넓혀 다시 시도해 주세요.")
            st.stop()

        st.session_state["candidate_cache_key"] = cache_key
        st.session_state["candidates"] = candidates

    with st.expander("🔎 이번 추천에 사용된 후보 정보(검증)"):
        st.write(f"- 후보 수: **{len(candidates)}개**")
        sample_df = pd.DataFrame(candidates[:20])
        cols = [c for c in ["name", "category", "address"] if c in sample_df.columns]
        st.dataframe(sample_df[cols], use_container_width=True, hide_index=True)

    with st.spinner("후보 중에서 최적의 장소를 고르는 중..."):
        try:
            r_data = recommend_from_candidates(client, payload, candidates)
        except Exception:
            st.error("추천 결과 생성에 실패했어요. (OpenAI 응답 파싱 실패)")
            st.stop()

    summary = r_data.get("summary", "추천 결과를 확인해 주세요.")
    recommendations = r_data.get("recommendations", [])

    if not isinstance(recommendations, list) or len(recommendations) == 0:
        st.error("추천 결과 형식이 올바르지 않습니다. 다시 시도해 주세요.")
        st.stop()

    recommendations = [r for r in recommendations if isinstance(r, dict)]
    recommendations = sorted(recommendations, key=lambda x: safe_int(x.get("rank", 999)))
    recommendations = recommendations[:TOP_K]
    recommendations = ensure_k_recommendations(recommendations, candidates, TOP_K)

    st.success(f"✅ **{summary}**")
    st.subheader(f"🏆 추천 TOP {TOP_K} (네이버 후보 기반)")

    for r in recommendations:
        name = r.get("name", "이름 없음")
        address = r.get("address", "") or "정보 없음"
        category = r.get("category", "") or "정보 없음"
        reason = r.get("reason", "")
        tags = r.get("tags", [])
        evidence = r.get("evidence", [])

        with st.container():
            left, right = st.columns([3, 2])

            with left:
                st.markdown(f"### {r.get('rank', '')}️⃣ {name}")
                if tags and isinstance(tags, list):
                    tag_line = " ".join([t if str(t).startswith("#") else f"#{t}" for t in tags[:10]])
                    st.caption(tag_line)
                st.write(f"📌 **추천 이유**: {reason}")
                st.write(f"🏷️ **카테고리**: {category}")
                st.write(f"📍 **주소**: {address}")

            with right:
                st.link_button("🗺️ 네이버 지도에서 보기", naver_map_search_url(name, address))
                if r.get("link"):
                    st.link_button("🔗 네이버/예약 링크", r["link"])

            if evidence and isinstance(evidence, list):
                with st.expander("🧾 추천 근거(요약)"):
                    for ev in evidence[:8]:
                        if ev:
                            st.write(f"- {ev}")

            if show_reviews:
                q = make_review_query(name, r.get("address", ""))
                with st.expander("🖼️ 블로그 후기 보기"):
                    st.caption(f"검색어: {q} | 정렬: {blog_sort}")
                    try:
                        blog_posts = naver_blog_search_cached(
                            q, naver_client_id, naver_client_secret,
                            display=review_display,
                            sort=blog_sort_param,
                        )
                    except Exception:
                        blog_posts = []
                        st.write("후기 검색에 실패했어요. 잠시 후 다시 시도해 주세요.")

                    if not blog_posts:
                        st.write("관련 블로그 후기를 찾지 못했어요.")
                    else:
                        for p in blog_posts[:review_display]:
                            cols = st.columns([1, 3])
                            with cols[0]:
                                if p.get("thumbnail"):
                                    st.image(p["thumbnail"], use_container_width=True)
                            with cols[1]:
                                st.markdown(f"- [{p['title']}]({p['link']})")
                                if p.get("desc"):
                                    st.caption(p["desc"])

            st.divider()

    st.subheader("📋 추천 결과 요약표")
    df = pd.DataFrame(recommendations)
    cols = [c for c in ["rank", "name", "category", "address", "link"] if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)

else:
    st.info("👆 조건을 입력하고 **추천 받기** 버튼을 눌러주세요.")
