import json
import re
import time
import html
from typing import List, Dict, Any, Tuple
from urllib.parse import quote_plus

import requests
import streamlit as st
import pandas as pd
from openai import OpenAI


# ===============================
# 기본 설정
# ===============================
DEFAULT_PEOPLE = 2
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
CANDIDATE_LIMIT_FOR_LLM = 35
REQUEST_SLEEP_SEC = 0.08

# 스코어링/검증 튜닝
SCORE_CANDIDATE_POOL = 60          # 스코어링에 사용할 최대 후보 수(많을수록 느림)
BLOG_AUGMENT_TOP_M = 18            # 블로그 스니펫으로 추가 점수 줄 후보 상위 M개
BLOG_SCORE_DISPLAY = 3             # 스코어링용 블로그 포스트 수(1~5 권장)


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


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def contains_any(text: str, keywords: List[str]) -> bool:
    t = text or ""
    return any(k in t for k in keywords)


def count_any(text: str, keywords: List[str]) -> int:
    t = text or ""
    return sum(1 for k in keywords if k in t)


def naver_local_search(
    query: str,
    client_id: str,
    client_secret: str,
    display: int = 5,
    sort: str = "comment",
    start: int = 1,
) -> List[Dict[str, str]]:
    """
    네이버 지역검색 API
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
        key = (normalize(c.get("name", "")), normalize(c.get("address", "")))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def filter_candidates(candidates: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    명백히 비식음료 업종 제거(병원/학원/부동산 등)
    """
    bad_keywords = [
        "학원", "공인중개", "부동산", "미용", "네일", "피부", "성형",
        "헬스", "요가", "필라테스", "세탁", "수리", "정비", "렌탈",
        "교회", "성당", "절", "약국", "병원", "의원", "치과", "한의원",
        "주유", "자동차", "인테리어", "가구", "마트",
    ]
    out = []
    for c in candidates:
        blob = f"{c.get('name','')} {c.get('category','')}"
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
    def _key(name: str, address: str) -> tuple:
        return (normalize(name), normalize(address))

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
# (1) Intent 추출 (LLM)
# ===============================
INTENT_LABELS = [
    "혼밥/1인식사",
    "빠른 이용",
    "모임/회식",
    "데이트/분위기",
    "카페/카공",
    "해장",
    "술자리/안주",
    "일반",
]


def infer_intent(client: OpenAI, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    사용자의 상황을 구조적 기준으로 변환(모델 출력은 '판정'에 사용, 최종 강제는 룰/스코어가 담당)
    """
    system = (
        "너는 음식점/카페 추천을 위한 '의도(intent) 판정기'다.\n"
        "출력은 JSON만.\n"
        "스키마:\n"
        "{\n"
        "  \"intent\": \"혼밥/1인식사|빠른 이용|모임/회식|데이트/분위기|카페/카공|해장|술자리/안주|일반\",\n"
        "  \"must_include\": [\"...\"] ,\n"
        "  \"must_exclude\": [\"...\"] ,\n"
        "  \"notes\": \"판정 근거 한 줄\"\n"
        "}\n"
        "- must_include/must_exclude는 업종/키워드 중심으로 짧게.\n"
        "- 모르면 intent는 '일반'으로.\n"
        "- 숫자/사실을 지어내지 마라."
    )

    food = payload.get("food_type") or []
    food_str = ", ".join(food) if food else "(선택 없음)"

    user = (
        f"visit_type(시간대/종류): {payload.get('visit_type')}\n"
        f"출발지: {payload.get('start_location') or '(미입력)'}\n"
        f"상황: {payload.get('situation')}\n"
        f"인원: {payload.get('people')}\n"
        f"이동거리: {payload.get('distance_pref')}\n"
        f"선호 종류: {food_str}\n"
        f"제외: {payload.get('exclude') or '(없음)'}\n"
        f"선호: {payload.get('prefer') or '(없음)'}\n\n"
        f"가능한 intent 라벨: {', '.join(INTENT_LABELS)}"
    )

    data = llm_json(client, system, user)
    intent = data.get("intent", "일반")
    if intent not in INTENT_LABELS:
        intent = "일반"
    must_in = data.get("must_include", [])
    must_ex = data.get("must_exclude", [])
    if not isinstance(must_in, list):
        must_in = []
    if not isinstance(must_ex, list):
        must_ex = []
    return {
        "intent": intent,
        "must_include": [normalize(str(x)) for x in must_in if normalize(str(x))],
        "must_exclude": [normalize(str(x)) for x in must_ex if normalize(str(x))],
        "notes": normalize(str(data.get("notes", ""))),
    }


# ===============================
# (2) Intent별 룰/스코어 함수
# ===============================
RULES = {
    "혼밥/1인식사": {
        "hard_exclude": ["무한리필", "단체", "연회", "웨딩", "대관", "뷔페"],
        "penalty": {
            "고기": -50, "구이": -50, "삼겹": -50, "갈비": -50, "양꼬치": -45,
            "주점": -60, "술집": -60, "호프": -60, "포차": -60, "바": -45,
        },
        "bonus": {
            "국밥": +25, "라멘": +25, "우동": +20, "덮밥": +20, "백반": +20,
            "분식": +18, "김밥": +18, "샐러드": +18, "버거": +15, "쌀국수": +18,
            "초밥": +12, "돈까스": +12, "제육": +10,
        },
        "blog_bonus_keywords": ["혼밥", "혼자", "1인", "점심", "런치", "점심특선", "회전율", "빠르게"],
    },
    "빠른 이용": {
        "hard_exclude": ["코스", "오마카세", "뷔페", "대관"],
        "penalty": {"웨이팅": -20, "주점": -30, "포차": -30, "바": -20},
        "bonus": {"분식": +20, "김밥": +18, "국밥": +18, "라멘": +15, "버거": +15, "샐러드": +12},
        "blog_bonus_keywords": ["회전율", "빨리", "금방", "대기", "포장", "키오스크"],
    },
    "모임/회식": {
        "hard_exclude": ["1인", "혼밥"],
        "penalty": {"카공": -10},
        "bonus": {"단체": +18, "룸": +18, "고기": +12, "구이": +12, "전골": +10, "한정식": +12},
        "blog_bonus_keywords": ["룸", "단체", "회식", "모임", "예약", "넓", "단체석"],
    },
    "데이트/분위기": {
        "hard_exclude": ["셀프", "푸드코트"],
        "penalty": {"분식": -8, "패스트푸드": -8},
        "bonus": {"와인": +12, "파스타": +12, "브런치": +10, "디저트": +10, "카페": +8},
        "blog_bonus_keywords": ["분위기", "데이트", "감성", "조명", "인테리어", "사진", "뷰"],
    },
    "카페/카공": {
        "hard_exclude": ["주점", "호프", "포차", "고기", "구이"],
        "penalty": {"식당": -5},
        "bonus": {"카페": +30, "디저트": +20, "베이커리": +18, "커피": +15},
        "blog_bonus_keywords": ["카공", "노트북", "콘센트", "조용", "좌석", "공부", "와이파이", "디저트"],
    },
    "해장": {
        "hard_exclude": ["디저트", "케이크"],
        "penalty": {"카페": -10},
        "bonus": {"국밥": +25, "해장": +20, "순대": +15, "감자탕": +18, "칼국수": +15, "라멘": +10},
        "blog_bonus_keywords": ["해장", "국물", "시원", "얼큰", "속풀이"],
    },
    "술자리/안주": {
        "hard_exclude": ["키즈", "학원"],
        "penalty": {"샐러드": -8},
        "bonus": {"주점": +25, "술집": +25, "호프": +20, "포차": +20, "바": +15, "안주": +12},
        "blog_bonus_keywords": ["안주", "분위기", "술", "2차", "맥주", "하이볼", "와인"],
    },
    "일반": {
        "hard_exclude": [],
        "penalty": {},
        "bonus": {},
        "blog_bonus_keywords": [],
    },
}


def score_candidate(
    c: Dict[str, str],
    intent: str,
    extra_must_exclude: List[str],
) -> Tuple[int, List[str]]:
    """
    후보 1개를 (룰 기반) 점수화. 점수+간단한 사유 로그 반환.
    """
    rule = RULES.get(intent, RULES["일반"])
    name = normalize(c.get("name", ""))
    cat = normalize(c.get("category", ""))
    blob = f"{name} {cat}"

    score = 0
    reasons = []

    # hard exclude
    hard_ex = rule.get("hard_exclude", []) + (extra_must_exclude or [])
    if hard_ex and contains_any(blob, hard_ex):
        return -10_000, ["하드 제외 키워드 매칭"]

    # penalty/bonus
    for k, v in (rule.get("penalty", {}) or {}).items():
        if k in blob:
            score += int(v)
            reasons.append(f"패널티:{k}{v}")

    for k, v in (rule.get("bonus", {}) or {}).items():
        if k in blob:
            score += int(v)
            reasons.append(f"보너스:{k}+{v}")

    # category 기반 약한 가점: 카페/디저트 선택했는데 관련 업종이면 가점
    if "카페" in blob:
        score += 6
    if "디저트" in blob or "베이커리" in blob:
        score += 4

    return score, reasons


def augment_score_with_blog_snippet(
    c: Dict[str, str],
    base_score: int,
    intent: str,
    naver_client_id: str,
    naver_client_secret: str,
    sort_param: str,
) -> Tuple[int, List[str]]:
    """
    블로그 스니펫(desc) 키워드 기반 가점
    - 호출 비용이 있으므로 상위 일부 후보에만 적용하도록 바깥에서 제어
    """
    rule = RULES.get(intent, RULES["일반"])
    kws = rule.get("blog_bonus_keywords", []) or []
    if not kws:
        return base_score, []

    q = make_review_query(c.get("name", ""), c.get("address", ""))
    try:
        posts = naver_blog_search_cached(
            q,
            naver_client_id,
            naver_client_secret,
            display=BLOG_SCORE_DISPLAY,
            sort=sort_param,
        )
    except Exception:
        return base_score, ["블로그 조회 실패(스코어 반영X)"]

    blob = " ".join([normalize(p.get("desc", "")) for p in posts if isinstance(p, dict)])
    hit = count_any(blob, kws)

    # 가점은 너무 세면 편향되므로 완만하게
    bonus = min(30, hit * 6)  # 키워드 1개당 +6, 최대 +30
    if bonus > 0:
        return base_score + bonus, [f"블로그키워드매칭 {hit}개(+{bonus})"]
    return base_score, []


def rank_candidates_with_rules(
    candidates: List[Dict[str, str]],
    intent_pack: Dict[str, Any],
    naver_client_id: str,
    naver_client_secret: str,
    blog_sort_param: str,
) -> List[Dict[str, Any]]:
    """
    (a) 룰 기반 1차 스코어링
    (b) 상위 일부만 블로그 스니펫으로 추가 가점
    (c) 최종 정렬 후 반환(원본 필드 + score 포함)
    """
    intent = intent_pack.get("intent", "일반")
    extra_ex = intent_pack.get("must_exclude", []) or []

    pool = candidates[:SCORE_CANDIDATE_POOL]

    scored = []
    for c in pool:
        s, logs = score_candidate(c, intent=intent, extra_must_exclude=extra_ex)
        scored.append({**c, "score": s, "_logs": logs})

    scored.sort(key=lambda x: safe_int(x.get("score", -999999), -999999), reverse=True)

    # 블로그 스니펫 가점(상위 M개만)
    top_m = scored[:min(BLOG_AUGMENT_TOP_M, len(scored))]
    rest = scored[min(BLOG_AUGMENT_TOP_M, len(scored)):]
    boosted = []

    for item in top_m:
        s0 = safe_int(item.get("score", 0), 0)
        s1, b_logs = augment_score_with_blog_snippet(
            item, s0, intent=intent,
            naver_client_id=naver_client_id,
            naver_client_secret=naver_client_secret,
            sort_param=blog_sort_param,
        )
        item["score"] = s1
        item["_logs"] = (item.get("_logs", []) or []) + b_logs
        boosted.append(item)

    boosted.sort(key=lambda x: safe_int(x.get("score", -999999), -999999), reverse=True)
    final_scored = boosted + rest
    final_scored.sort(key=lambda x: safe_int(x.get("score", -999999), -999999), reverse=True)
    return final_scored


# ===============================
# (3) 스코어 상위 후보만 LLM에 넘겨 추천
# ===============================
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

    # 출발지 강제 포함(가능하면)
    start = (payload.get("start_location") or "").strip()
    if start:
        patched = []
        for q in queries:
            patched.append(q if start in q else f"{start} {q}".strip())
        queries = patched

    # 중복 제거
    uniq = []
    seen = set()
    for q in queries:
        if q in seen:
            continue
        seen.add(q)
        uniq.append(q)
    return uniq[:MAX_QUERIES]


def collect_candidates(
    queries: List[str],
    naver_client_id: str,
    naver_client_secret: str,
) -> List[Dict[str, str]]:
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


def recommend_from_candidates(
    client: OpenAI,
    payload: Dict[str, Any],
    intent_pack: Dict[str, Any],
    ranked_candidates: List[Dict[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    """
    스코어 상위 후보만 LLM에 넘겨 최종 선정
    """
    intent = intent_pack.get("intent", "일반")
    must_ex = intent_pack.get("must_exclude", []) or []
    must_in = intent_pack.get("must_include", []) or []

    system_rec = (
        "너는 음식점/카페 추천 큐레이터다.\n"
        "- 반드시 candidates 목록에 있는 장소만 추천할 수 있다.\n"
        "- candidates에 없는 장소를 새로 만들면 실패다.\n"
        "- 사용자의 intent(의도)와 MUST 조건을 우선으로 지켜라.\n"
        "- MUST_EXCLUDE 조건에 걸리는 장소는 추천하지 마라.\n"
        "- 숫자(평점/가격/거리/시간)는 근거 데이터가 없으면 절대 지어내지 마라.\n"
        "- 출력은 JSON만.\n"
        "스키마:\n"
        "{\n"
        "  \"summary\": \"한 줄 결론(숫자/개수 언급 금지)\",\n"
        "  \"recommendations\": [\n"
        "    {\n"
        "      \"rank\": 1,\n"
        "      \"name\": \"...\",\n"
        "      \"reason\": \"사용자 의도에 맞는 이유(짧고 명확)\",\n"
        "      \"tags\": [\"#키워드\", \"#키워드\"],\n"
        "      \"address\": \"...\",\n"
        "      \"category\": \"...\",\n"
        "      \"link\": \"...\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        f"- recommendations는 가능한 한 정확히 {top_k}개를 채워라.\n"
        "- rank는 1부터 연속.\n"
        "- summary에는 '3곳/5곳' 같은 개수 표현을 쓰지 마라."
    )

    llm_candidates = []
    for c in ranked_candidates[:CANDIDATE_LIMIT_FOR_LLM]:
        llm_candidates.append({
            "name": c.get("name", ""),
            "address": c.get("address", ""),
            "category": c.get("category", ""),
            "link": c.get("link", ""),
            "score_hint": c.get("score", 0),  # 참고용(모델이 숫자 근거로 쓰지 않게)
        })

    llm_payload = {
        "intent": intent,
        "must_include": must_in,
        "must_exclude": must_ex,
        "visit_type": payload.get("visit_type", "상관없음"),
        "start_location": payload.get("start_location", ""),
        "situation": payload.get("situation", ""),
        "people": payload.get("people", 0),
        "distance_pref": payload.get("distance_pref", ""),
        "food_type": payload.get("food_type", []),
        "exclude": payload.get("exclude", ""),
        "prefer": payload.get("prefer", ""),
        "top_k": top_k,
        "candidates": llm_candidates,
    }
    user_rec = json.dumps(llm_payload, ensure_ascii=False)
    return llm_json(client, system_rec, user_rec)


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="LunchMate 🍱", layout="wide")

# 스크롤 잠김 방지 CSS
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
if "ranked_candidates" not in st.session_state:
    st.session_state["ranked_candidates"] = []
if "intent_pack" not in st.session_state:
    st.session_state["intent_pack"] = None


def require_secrets_or_stop():
    if not (naver_client_id and naver_client_secret and openai_api_key):
        st.error("서비스 설정 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")
        st.stop()


# ===============================
# 사이드바
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
people = st.sidebar.slider("인원 수", 1, 10, DEFAULT_PEOPLE)
distance = st.sidebar.selectbox("이동 거리", DISTANCE_OPTIONS, index=DEFAULT_DISTANCE_INDEX)
food_type = st.sidebar.multiselect("음식/카페 종류", FOOD_OPTIONS, default=DEFAULT_FOOD_TYPES)

st.sidebar.header("🚫 제외 / ✅ 선호")
exclude_text = st.sidebar.text_input("제외 조건(쉼표로 구분)", placeholder="예: 매운 음식, 회, 웨이팅")
prefer_text = st.sidebar.text_input("선호 조건(쉼표로 구분)", placeholder="예: 혼밥, 조용한 곳, 가성비, 디저트")

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
    placeholder="예: 점심에 혼밥하기 좋은 곳 / 회식하기 좋은 고깃집 / 카공 가능한 조용한 카페 등",
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("⚡ 빨리 이용"):
        situation = "시간이 없어서 빨리 이용할 수 있는 곳을 찾고 있어요"
with col2:
    if st.button("👥 모임/회식"):
        situation = "여럿이 조용히 대화할 수 있는 모임 장소가 필요해요"
with col3:
    if st.button("🥣 해장"):
        situation = "어제 술을 마셔서 해장에 좋은 음식을 먹고 싶어요"
with col4:
    if st.button("☕ 카페"):
        situation = "카공하기 좋고 콘센트/좌석이 괜찮은 카페를 찾고 있어요"

st.write("")

# 버튼(재추천)
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

    # reroll: 후보/스코어 재사용(의도는 다시 뽑아도 되지만, 기준 고정이 더 나아서 그대로 재사용)
    if reroll and st.session_state.get("candidates") and st.session_state.get("candidate_cache_key") == cache_key:
        candidates = st.session_state["candidates"]
        ranked_candidates = st.session_state.get("ranked_candidates", [])
        intent_pack = st.session_state.get("intent_pack") or {"intent": "일반", "must_include": [], "must_exclude": [], "notes": ""}
    else:
        with st.spinner("의도/기준을 정리하는 중..."):
            try:
                intent_pack = infer_intent(client, payload)
            except Exception:
                # 의도 추출이 실패해도 서비스는 계속 동작하게(보수적으로 '일반')
                intent_pack = {"intent": "일반", "must_include": [], "must_exclude": [], "notes": ""}

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
                candidates = collect_candidates(queries, naver_client_id, naver_client_secret)
            except requests.HTTPError as e:
                st.error(f"네이버 지역검색 API 호출 실패(HTTP): {e}")
                st.stop()
            except requests.RequestException as e:
                st.error(f"네트워크 오류: {e}")
                st.stop()

        if not candidates:
            st.warning("조건에 맞는 실제 후보를 찾지 못했어요. 키워드를 넓혀 다시 시도해 주세요.")
            st.stop()

        with st.spinner("후보를 '명확한 기준'으로 정렬하는 중..."):
            ranked_candidates = rank_candidates_with_rules(
                candidates=candidates,
                intent_pack=intent_pack,
                naver_client_id=naver_client_id,
                naver_client_secret=naver_client_secret,
                blog_sort_param=blog_sort_param,
            )

        st.session_state["candidate_cache_key"] = cache_key
        st.session_state["candidates"] = candidates
        st.session_state["ranked_candidates"] = ranked_candidates
        st.session_state["intent_pack"] = intent_pack

    # 검증용(원하면 숨겨도 됨)
    with st.expander("🔎 이번 추천에 사용된 기준/후보(검증)"):
        st.write(f"- 판정 intent: **{intent_pack.get('intent','일반')}**")
        if intent_pack.get("notes"):
            st.caption(f"판정 메모: {intent_pack.get('notes')}")
        if intent_pack.get("must_exclude"):
            st.write(f"- MUST_EXCLUDE: {', '.join(intent_pack.get('must_exclude'))}")
        st.write(f"- 후보 수: **{len(candidates)}개**")
        sample_df = pd.DataFrame(ranked_candidates[:25])
        cols = [c for c in ["score", "name", "category", "address"] if c in sample_df.columns]
        st.dataframe(sample_df[cols], use_container_width=True, hide_index=True)

    with st.spinner("스코어 상위 후보 중에서 최적의 장소를 고르는 중..."):
        try:
            r_data = recommend_from_candidates(client, payload, intent_pack, ranked_candidates, TOP_K)
        except Exception:
            st.error("추천 결과 생성에 실패했어요. (OpenAI 응답 파싱 실패)")
            st.stop()

    # ✅ summary는 개수 혼선 방지를 위해 고정 문구 사용
    fixed_summary = f"조건에 맞는 추천 TOP {TOP_K} 결과입니다."
    recommendations = r_data.get("recommendations", [])

    if not isinstance(recommendations, list) or len(recommendations) == 0:
        st.error("추천 결과 형식이 올바르지 않습니다. 다시 시도해 주세요.")
        st.stop()

    recommendations = [r for r in recommendations if isinstance(r, dict)]
    recommendations = sorted(recommendations, key=lambda x: safe_int(x.get("rank", 999)))
    recommendations = recommendations[:TOP_K]
    recommendations = ensure_k_recommendations(recommendations, ranked_candidates, TOP_K)

    st.success(f"✅ **{fixed_summary}**")
    st.subheader(f"🏆 추천 TOP {TOP_K} (네이버 후보 기반)")

    for r in recommendations:
        name = r.get("name", "이름 없음")
        address = r.get("address", "") or "정보 없음"
        category = r.get("category", "") or "정보 없음"
        reason = r.get("reason", "")
        tags = r.get("tags", [])

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
