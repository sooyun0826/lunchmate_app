import json
import re
import time
import html
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import quote_plus

import requests
import streamlit as st
import pandas as pd
from openai import OpenAI


# ===============================
# 기본 디폴트 설정 (초기값)
# ===============================
# ✅ 인원수 디폴트: '상관없음'
PEOPLE_OPTIONS = ["상관없음"] + [f"{i}명" for i in range(1, 11)]
DEFAULT_PEOPLE_INDEX = 0

DISTANCE_OPTIONS = ["5분 이내", "10분 이내", "상관없음"]
DEFAULT_DISTANCE_INDEX = 2  # "상관없음"

FOOD_OPTIONS = [
    "한식", "중식", "일식", "양식", "분식", "기타",
    "카페", "디저트"
]
DEFAULT_FOOD_TYPES: List[str] = []

TOP_K = 5

# 후보 확장 / 성능
MAX_QUERIES = 6
LOCAL_DISPLAY_PER_QUERY = 5
CANDIDATE_LIMIT_FOR_LLM = 40
REQUEST_SLEEP_SEC = 0.08

# 블로그 분석용(스니펫만 사용)
BLOG_PER_PLACE_FOR_SCORING = 3

# 스코어링 이후 LLM에 넘길 후보 수
LLM_RERANK_POOL = 25

# ✅ 결과 카드 이미지(이미지 검색 API)
IMAGE_PER_PLACE = 1  # 1~5 (네이버 이미지 API display 제한에 맞춰 5 이하로 유지)


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


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def split_csv(text: str) -> List[str]:
    parts = [p.strip() for p in (text or "").split(",")]
    return [p for p in parts if p]


def any_kw(blob: str, kws: List[str]) -> bool:
    b = normalize_text(blob)
    return any(normalize_text(k) in b for k in kws if k)


def count_kw_hits(blob: str, kws: List[str]) -> int:
    b = normalize_text(blob)
    hits = 0
    for k in kws:
        kk = normalize_text(k)
        if kk and kk in b:
            hits += 1
    return hits


def parse_people_value(choice: str) -> int:
    """'상관없음' -> 0, 'N명' -> N"""
    choice = (choice or "").strip()
    if choice == "상관없음":
        return 0
    m = re.search(r"(\d+)", choice)
    return int(m.group(1)) if m else 0


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
        key = ((c.get("name", "") or "").strip(), (c.get("address", "") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def filter_candidates(candidates: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    음식점/카페 추천 서비스 기준으로, 명백히 비식음료 업종만 제거
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

    candidate = text[start: end + 1]
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
    candidates: List[Dict[str, Any]],
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
                "tags": ["후보기반"],
                "address": c.get("address", ""),
                "category": c.get("category", ""),
                "link": c.get("link", ""),
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
        })
    return items


@st.cache_data(ttl=3600, show_spinner=False)
def naver_image_search_cached(
    query: str,
    client_id: str,
    client_secret: str,
    display: int = 1,
    sort: str = "sim",
):
    """
    네이버 이미지 검색 API
    https://openapi.naver.com/v1/search/image
    - items[].thumbnail : 섬네일 URL
    - items[].link      : 원본 이미지 URL
    """
    url = "https://openapi.naver.com/v1/search/image"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    params = {
        "query": query,
        "display": max(1, min(display, 5)),
        "start": 1,
        "sort": sort,        # sim/date
        "filter": "all",     # all/large/medium/small
    }
    r = requests.get(url, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get("items", [])


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
        "blog_sort": payload.get("blog_sort", "sim"),
        "quick_tags": payload.get("quick_tags", []),
    }
    return json.dumps(compact, ensure_ascii=False, sort_keys=True)


# ===============================
# “명확한 기준” 룰셋 (혼밥 강화)
# ===============================
SOLO_HARD_EXCLUDE = ["웨딩", "대관", "연회", "행사", "세미나", "뷔페", "돌잔치"]
SOLO_STRONG_PENALTY = [
    "단체", "단체석", "단체가능", "회식", "모임", "룸", "룸완비", "가족모임",
    "예약필수", "대형", "연말",
]
SOLO_POSITIVE = ["혼밥", "혼자", "1인", "1인식사", "바자리", "카운터", "키오스크", "포장", "테이크아웃"]
SOLO_CATEGORY_PENALTY = ["고기", "삼겹", "갈비", "한우", "무한리필", "바베큐", "곱창", "막창", "참치", "횟집", "대게", "코스", "뷔페"]

BUDGET_POSITIVE = ["가성비", "저렴", "착한가격", "만원", "만원대", "점심특선", "세트", "백반"]
BUDGET_NEGATIVE = ["오마카세", "파인다이닝", "코스", "프리미엄", "고급", "비싼", "고가"]


def infer_intents(payload: Dict[str, Any]) -> Dict[str, bool]:
    situation = payload.get("situation", "")
    prefer = payload.get("prefer", "")
    people = payload.get("people", 0)

    blob = f"{situation} {prefer}".strip()
    solo = (people == 1) or any_kw(blob, ["혼밥", "혼자", "1인", "1인식사", "혼술"])
    budget = any_kw(blob, ["가성비", "저렴", "싸게", "착한가격", "만원", "만원대"])
    vegan = any_kw(blob, ["비건", "vegan", "채식", "락토", "오보"])
    diet = any_kw(blob, ["다이어트", "저탄", "키토", "샐러드", "단백질"])
    return {"solo": solo, "budget": budget, "vegan": vegan, "diet": diet}


def candidate_signal_blob(candidate: Dict[str, str], blog_snippets: List[str]) -> str:
    name = candidate.get("name", "") or ""
    category = candidate.get("category", "") or ""
    addr = candidate.get("address", "") or ""
    blog = " ".join(blog_snippets[:10])
    return f"{name} {category} {addr} {blog}".strip()


def score_candidate_for_payload(
    payload: Dict[str, Any],
    candidate: Dict[str, str],
    blog_snippets: List[str],
    intents: Dict[str, bool],
) -> Tuple[int, Dict[str, Any]]:
    score = 0
    reasons = []

    blob = candidate_signal_blob(candidate, blog_snippets)
    name_cat = f"{candidate.get('name','')} {candidate.get('category','')}".strip()

    exclude_terms = split_csv(payload.get("exclude", ""))
    if exclude_terms and any_kw(blob, exclude_terms):
        score -= 120
        reasons.append(f"제외조건 매칭(-120): {', '.join(exclude_terms)}")

    if intents.get("solo"):
        if any_kw(blob, SOLO_HARD_EXCLUDE):
            score -= 999
            reasons.append("혼밥: 하드 제외 용도/업종(-999)")
        hits = count_kw_hits(blob, SOLO_STRONG_PENALTY)
        if hits:
            penalty = min(80 * hits, 240)
            score -= penalty
            reasons.append(f"혼밥: 단체/모임 시그널({hits}) 감점(-{penalty})")

        pos_hits = count_kw_hits(blob, SOLO_POSITIVE)
        if pos_hits:
            bonus = min(50 * pos_hits, 150)
            score += bonus
            reasons.append(f"혼밥: 1인 친화 시그널({pos_hits}) 가점(+{bonus})")

        cat_hits = count_kw_hits(name_cat, SOLO_CATEGORY_PENALTY)
        if cat_hits:
            penalty = min(70 * cat_hits, 210)
            score -= penalty
            reasons.append(f"혼밥: 업종 패널티({cat_hits})(-{penalty})")

    if intents.get("budget"):
        pos = count_kw_hits(blob, BUDGET_POSITIVE)
        if pos:
            bonus = min(35 * pos, 140)
            score += bonus
            reasons.append(f"가성비: 긍정 시그널({pos})(+{bonus})")
        neg = count_kw_hits(blob, BUDGET_NEGATIVE)
        if neg:
            penalty = min(60 * neg, 180)
            score -= penalty
            reasons.append(f"가성비: 고가 시그널({neg})(-{penalty})")

    # ✅ 빠른 태그 반영(약~중간)
    quick_tags = payload.get("quick_tags", []) or []
    if quick_tags:
        if any_kw(blob, quick_tags):
            score += 40
            reasons.append("빠른 태그 매칭(+40)")

    food_types = payload.get("food_type") or []
    if food_types:
        if any_kw(candidate.get("category", ""), food_types) or any_kw(candidate.get("name", ""), food_types):
            score += 35
            reasons.append("선택한 음식/카페 종류 매칭(+35)")

    visit = payload.get("visit_type", "상관없음")
    if visit == "카페/디저트":
        if any_kw(blob, ["카페", "디저트", "베이커리", "커피"]):
            score += 40
            reasons.append("방문목적(카페/디저트) 매칭(+40)")
        else:
            score -= 20
            reasons.append("방문목적(카페/디저트) 불일치(-20)")
    elif visit in ["아침", "점심", "저녁"]:
        if any_kw(blob, ["브런치", "아침"]):
            score += 15 if visit == "아침" else 0
        if any_kw(blob, ["점심특선", "런치"]):
            score += 15 if visit == "점심" else 0
        if any_kw(blob, ["술", "안주", "호프", "포차"]):
            score += 15 if visit == "저녁" else 0

    score += 10  # 네이버 후보 기반 신뢰 가점

    meta = {"score": score, "score_notes": reasons[:8]}
    return score, meta


def solo_gate_filter(sorted_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept = []
    dropped = []
    penalty_set = [normalize_text(x) for x in SOLO_STRONG_PENALTY]
    for c in sorted_candidates:
        blob = normalize_text(c.get("_signal_blob", ""))
        if any(k in blob for k in penalty_set):
            dropped.append(c)
        else:
            kept.append(c)

    if len(kept) >= 15:
        return kept

    dropped = sorted(dropped, key=lambda x: safe_int(x.get("_score", -999999)), reverse=True)
    return kept + dropped[: max(0, 15 - len(kept))]


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="LunchMate 🍱", layout="wide")

# ✅ 스크롤 잠김 방지 CSS
st.markdown(
    """
    <style>
    html, body { overflow: auto !important; height: auto !important; }
    [data-testid="stAppViewContainer"] { overflow: auto !important; }
    [data-testid="stMain"] { overflow: auto !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🍽️ LunchMate 🍽️")
st.caption(f"사용자님의 상황과 선호도를 바탕으로 음식점/카페 후보 중 최적의 {TOP_K}곳을 추천해 드립니다.")

naver_client_id = get_secret("NAVER_CLIENT_ID")
naver_client_secret = get_secret("NAVER_CLIENT_SECRET")
openai_api_key = get_secret("OPENAI_API_KEY")

if "candidate_cache_key" not in st.session_state:
    st.session_state["candidate_cache_key"] = None
if "candidates" not in st.session_state:
    st.session_state["candidates"] = []
# ✅ 메인에 위치한 빠른 태그 상태 유지
if "quick_tags_main" not in st.session_state:
    st.session_state["quick_tags_main"] = []


# ===============================
# 사이드바 (✅ 빠른 태그 제거!)
# ===============================
st.sidebar.header("🕒 매장 방문 목적")
visit_type = st.sidebar.selectbox(
    "추천 받을 종류",
    ["상관없음", "아침", "점심", "저녁", "카페/디저트"],
    index=0
)

st.sidebar.header("📍 출발 위치(정확도 개선)")
start_location = st.sidebar.text_input("출발지(회사/역/주소)", placeholder="예: 신촌역, 강남역, 판교역")

st.sidebar.header("🔍 검색 조건")
people_choice = st.sidebar.selectbox("인원 수", PEOPLE_OPTIONS, index=DEFAULT_PEOPLE_INDEX)
people = parse_people_value(people_choice)

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

st.sidebar.divider()
debug_mode = st.sidebar.checkbox("🧪 디버그(후보 점수/필터 보기)", value=False)


# ===============================
# 메인 입력 (✅ 빠른 태그를 프롬프트 바로 아래로 이동)
# ===============================
st.subheader("📝 희망 조건을 자유롭게 입력해 주세요")
situation = st.text_area(
    "자유롭게 상황을 입력해 주세요(취향, 방문 지역, 인원 수, 식사 상황 등)",
    placeholder="예: 신촌역에서 친구와 점심 먹을거야. 가성비 좋은 중식 음식점 추천해줘. / 잠실에서 카공하기 좋은 카페 찾아줘.",
)

# ✅ 프롬프트 입력란 바로 아래에 '빠른 태그' 배치
st.markdown("### 🧩 빠른 태그(복수 선택 가능)")
QUICK_TAGS = [
    "혼밥", "조용한", "가성비", "웨이팅 적은", "매운 음식",
    "데이트", "단체 가능", "포장/테이크아웃",
    "다이어트", "비건", "샐러드", "디저트", "브런치",
    "야식", "술/안주", "카공",
]
quick_tags = st.multiselect(
    "원하는 키워드를 선택하세요",
    QUICK_TAGS,
    default=st.session_state.get("quick_tags_main", []),
    key="quick_tags_main",
)

# ✅ 적용 표시(사용자 가시성)
if quick_tags:
    st.success(f"✅ 빠른 태그 적용됨: {', '.join(quick_tags)}")
else:
    st.caption("선택한 빠른 태그가 없어요. 필요하면 위에서 골라주세요.")

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
        "점심": "점심",
        "저녁": "저녁/술자리",
        "카페/디저트": "카페/디저트",
        "상관없음": "맛집/카페",
    }.get(visit, "맛집/카페")

    qt = payload.get("quick_tags") or []
    qt_str = ", ".join(qt) if qt else "(없음)"

    user_query = (
        f"추천 종류: {visit} ({visit_hint})\n"
        f"출발지: {payload.get('start_location') or '(미입력)'}\n"
        f"상황: {payload.get('situation')}\n"
        f"인원: {payload.get('people') or '상관없음'}\n"
        f"이동거리 선호: {payload.get('distance_pref')}\n"
        f"선호 음식/카페 종류: {food_str}\n"
        f"빠른 태그: {qt_str}\n"
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


def score_and_prepare_candidates(
    payload: Dict[str, Any],
    candidates: List[Dict[str, str]],
    blog_sort_param: str,
) -> List[Dict[str, Any]]:
    intents = infer_intents(payload)

    enriched: List[Dict[str, Any]] = []
    for c in candidates:
        name = c.get("name", "")
        addr = c.get("address", "")
        q = make_review_query(name, addr)

        try:
            posts = naver_blog_search_cached(
                q, naver_client_id, naver_client_secret,
                display=BLOG_PER_PLACE_FOR_SCORING,
                sort=blog_sort_param,
            )
            snippets = []
            for p in posts[:BLOG_PER_PLACE_FOR_SCORING]:
                snippets.append(f"{p.get('title','')} {p.get('desc','')}".strip())
        except Exception:
            snippets = []

        score, meta = score_candidate_for_payload(payload, c, snippets, intents)
        signal_blob = candidate_signal_blob(c, snippets)

        c2 = dict(c)
        c2["_score"] = score
        c2["_score_notes"] = meta.get("score_notes", [])
        c2["_signal_blob"] = signal_blob
        enriched.append(c2)

    enriched = sorted(enriched, key=lambda x: safe_int(x.get("_score", -999999)), reverse=True)

    if intents.get("solo"):
        enriched = solo_gate_filter(enriched)

    return enriched


def recommend_from_candidates(
    client: OpenAI,
    payload: Dict[str, Any],
    candidates_for_llm: List[Dict[str, Any]],
) -> Dict[str, Any]:
    intents = infer_intents(payload)
    must_notes = []
    if intents.get("solo"):
        must_notes.append("- 사용자가 혼밥/1인식사를 원한다. 단체/회식/모임 중심 장소는 우선 배제하라.")
        must_notes.append("- 바자리/카운터/1인식사 힌트가 있거나 혼자 먹기 편한 형태를 우선하라.")
    if intents.get("budget"):
        must_notes.append("- 사용자가 가성비/저렴을 원한다. '오마카세/파인다이닝/고급/코스' 느낌은 배제하라.")
    if intents.get("vegan"):
        must_notes.append("- 사용자가 비건/채식을 원한다. 고기 중심 식당은 배제하고 채식 선택지가 있는 후보를 우선하라.")
    if intents.get("diet"):
        must_notes.append("- 사용자가 다이어트를 원한다. 샐러드/단백질/저탄수 옵션이 가능한 후보를 우선하라.")

    must_notes_text = "\n".join(must_notes) if must_notes else "- 사용자 조건에 가장 부합하는 후보를 우선하라."

    system_rec = (
        "너는 음식점/카페 추천 큐레이터다.\n"
        "- 반드시 candidates 목록에 있는 장소만 추천할 수 있다.\n"
        "- candidates에 없는 장소를 새로 만들면 실패다.\n"
        "- 숫자(평점/가격/거리/시간)는 근거 데이터가 없으면 절대 지어내지 마라.\n"
        "- 출력은 JSON만.\n"
        f"{must_notes_text}\n"
        "스키마:\n"
        "{\n"
        "  \"recommendations\": [\n"
        "    {\n"
        "      \"rank\": 1,\n"
        "      \"name\": \"...\",\n"
        "      \"reason\": \"...\",\n"
        "      \"tags\": [\"#가성비\", \"#혼밥\", \"#조용한\"],\n"
        "      \"address\": \"...\",\n"
        "      \"category\": \"...\",\n"
        "      \"link\": \"...\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        f"- recommendations는 반드시 정확히 {TOP_K}개를 출력하라.\n"
        "- rank는 1부터 연속이어야 한다.\n"
        "- tags는 짧게 2~5개."
    )

    slim_candidates = []
    for c in candidates_for_llm[:CANDIDATE_LIMIT_FOR_LLM]:
        slim_candidates.append({
            "name": c.get("name", ""),
            "address": c.get("address", ""),
            "category": c.get("category", ""),
            "link": c.get("link", ""),
            "signal": " ".join((c.get("_score_notes") or [])[:3]),
        })

    llm_payload = {
        "visit_type": payload.get("visit_type", "상관없음"),
        "start_location": payload.get("start_location", ""),
        "situation": payload.get("situation", ""),
        "people": payload.get("people", 0),
        "distance_pref": payload.get("distance_pref", ""),
        "food_type": payload.get("food_type", []),
        "quick_tags": payload.get("quick_tags", []),
        "exclude": payload.get("exclude", ""),
        "prefer": payload.get("prefer", ""),
        "top_k": TOP_K,
        "candidates": slim_candidates,
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
    if not situation.strip():
        st.warning("상황을 입력해 주세요.")
        st.stop()

    require_secrets_or_stop()
    client = OpenAI(api_key=openai_api_key)

    payload = {
        "visit_type": visit_type,
        "start_location": start_location.strip(),
        "situation": situation.strip(),
        "people": people,  # 0이면 상관없음
        "distance_pref": distance,
        "food_type": food_type,
        "quick_tags": quick_tags,  # ✅ 이제 메인 입력 아래에서 선택된 값
        "exclude": exclude_text.strip(),
        "prefer": prefer_text.strip(),
        "blog_sort": blog_sort_param,
    }
    cache_key = build_cache_key(payload)

    # 1) 후보 수집(캐시)
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

    # 2) 후보 스코어링 + 혼밥 게이트 필터
    with st.spinner("후보를 정교하게 선별하는 중..."):
        scored_candidates = score_and_prepare_candidates(payload, candidates, blog_sort_param)

    # (검증: 후보 보기)
    with st.expander("🔎 이번 추천에 사용된 후보 정보(검증)"):
        st.write(f"- 후보 수(원본): **{len(candidates)}개**")
        st.write(f"- 후보 수(스코어링/필터 후): **{len(scored_candidates)}개**")
        sample_df = pd.DataFrame(scored_candidates[:30])
        cols = [c for c in ["name", "category", "address", "_score"] if c in sample_df.columns]
        st.dataframe(sample_df[cols], use_container_width=True, hide_index=True)

        if debug_mode:
            st.caption("상위 후보 일부의 점수/판정 메모(디버그)")
            for c in scored_candidates[:10]:
                st.write(f"- **{c.get('name')}** ({c.get('_score')}): {c.get('_score_notes')}")

    # 3) LLM 최종 추천(후보 중 선택)
    with st.spinner("후보 중에서 최적의 장소를 고르는 중..."):
        try:
            pool = scored_candidates[:LLM_RERANK_POOL]
            r_data = recommend_from_candidates(client, payload, pool)
        except Exception:
            st.error("추천 결과 생성에 실패했어요. (OpenAI 응답 파싱 실패)")
            st.stop()

    recommendations = r_data.get("recommendations", [])
    if not isinstance(recommendations, list) or len(recommendations) == 0:
        st.error("추천 결과 형식이 올바르지 않습니다. 다시 시도해 주세요.")
        st.stop()

    recommendations = [r for r in recommendations if isinstance(r, dict)]
    recommendations = sorted(recommendations, key=lambda x: safe_int(x.get("rank", 999)))
    recommendations = recommendations[:TOP_K]
    recommendations = ensure_k_recommendations(recommendations, scored_candidates, TOP_K)

    fixed_summary = f"조건에 맞는 추천 TOP {TOP_K} 결과입니다."
    st.success(f"✅ **{fixed_summary}**")
    st.subheader(f"🏆 추천 TOP {TOP_K} (네이버 후보 기반)")

    for r in recommendations:
        name = r.get("name", "이름 없음")
        address = r.get("address", "") or ""
        category = r.get("category", "") or "정보 없음"
        reason = r.get("reason", "")
        tags = r.get("tags", [])

        # ✅ 이미지 검색: 매장명 + 주소 힌트(앞 1~2토큰)
        addr_hint = " ".join(address.split()[:2]).strip()
        img_query = f"{name} {addr_hint}".strip()

        img_items = []
        try:
            img_items = naver_image_search_cached(
                img_query, naver_client_id, naver_client_secret,
                display=IMAGE_PER_PLACE, sort="sim"
            )
        except Exception:
            img_items = []

        with st.container():
            img_col, info_col = st.columns([1, 2])

            with img_col:
                if img_items:
                    thumb = img_items[0].get("thumbnail")
                    if thumb:
                        st.image(thumb, use_container_width=True)
                        src = img_items[0].get("link")
                        if src:
                            st.link_button("이미지 출처", src)
                else:
                    st.caption("이미지를 찾지 못했어요.")

            with info_col:
                st.markdown(f"### {r.get('rank', '')}️⃣ {name}")
                if tags and isinstance(tags, list):
                    tag_line = " ".join([t if str(t).startswith("#") else f"#{t}" for t in tags[:10]])
                    st.caption(tag_line)
                st.write(f"📌 **추천 이유**: {reason}")
                st.write(f"🏷️ **카테고리**: {category}")
                st.write(f"📍 **주소**: {address or '정보 없음'}")

                st.link_button("🗺️ 네이버 지도에서 보기", naver_map_search_url(name, address))
                if r.get("link"):
                    st.link_button("🔗 네이버/예약 링크", r["link"])

            if show_reviews:
                q = make_review_query(name, address)
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
