import json

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

# -------------------------------
# 기본 설정
# -------------------------------
st.set_page_config(
    page_title="LunchMate 🍱",
    layout="wide"
)

st.title("🍽️ LunchMate")
st.caption("직장인의 상황과 선호도를 분석해 최적의 점심 식당 3곳을 추천합니다")

# -------------------------------
# Sidebar (검색 조건 필터)
# -------------------------------
st.sidebar.header("🔍 검색 조건")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="키를 입력하면 챗봇이 상황을 분석해 식당을 추천합니다.",
)

people = st.sidebar.slider("인원 수", 1, 10, 5)
distance = st.sidebar.selectbox("이동 거리", ["5분 이내", "10분 이내", "상관없음"])
food_type = st.sidebar.multiselect(
    "음식 종류",
    ["한식", "중식", "일식", "양식", "분식", "기타"],
    default=["한식"]
)

# -------------------------------
# 메인 입력 영역
# -------------------------------
st.subheader("📝 오늘의 상황을 입력해 주세요")

situation = st.text_area(
    "자연스럽게 입력해 주세요",
    placeholder="예: 오늘 팀장님 모시고 5명이서 조용히 1시간 안에 먹어야 해요"
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

# -------------------------------
# 추천 버튼
# -------------------------------
if st.button("🤖 점심 추천 받기") and situation:
    if not api_key:
        st.warning("사이드바에 OpenAI API Key를 입력해 주세요.")
        st.stop()

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "너는 직장인 점심 추천 챗봇이다. "
        "사용자의 상황과 선호 조건을 분석해 식당 3곳을 추천한다. "
        "각 추천에는 간단한 이유와 대략적인 평점, 거리(분), 가격(원)을 포함한다. "
        "응답은 반드시 JSON 형식으로만 출력한다."
    )
    user_prompt = (
        "아래 조건을 고려해 점심 식당 3곳을 추천해 줘.\n"
        f"- 상황: {situation}\n"
        f"- 인원 수: {people}\n"
        f"- 이동 거리 선호: {distance}\n"
        f"- 음식 종류 선호: {', '.join(food_type) if food_type else '상관없음'}\n\n"
        "출력 JSON 스키마:\n"
        "{\n"
        "  \"summary\": \"한 줄 결론\",\n"
        "  \"recommendations\": [\n"
        "    {\"rank\": 1, \"name\": \"식당명\", \"reason\": \"추천 이유\", \"rating\": 4.5, \"distance\": 5, \"price\": 12000}\n"
        "  ]\n"
        "}\n"
        "추천은 반드시 3개만 포함해."
    )

    with st.spinner("AI가 식당을 추천하는 중입니다..."):
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

    raw_output = response.output_text
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        st.error("AI 응답을 해석하지 못했습니다. 잠시 후 다시 시도해 주세요.")
        st.stop()

    recommendations = parsed.get("recommendations", [])
    summary = parsed.get("summary", "추천 결과를 확인해 주세요.")

    if len(recommendations) != 3:
        st.error("추천 결과가 3개가 아닙니다. 다시 시도해 주세요.")
        st.stop()

    df = pd.DataFrame(recommendations)

    # -------------------------------
    # 한 줄 결론
    # -------------------------------
    st.success(f"✅ **{summary}**")

    # -------------------------------
    # 추천 카드
    # -------------------------------
    st.subheader("🏆 추천 식당 TOP 3")

    for r in recommendations:
        with st.container():
            st.markdown(f"### {r['rank']}️⃣ {r['name']}")
            st.write(f"📌 추천 이유: {r['reason']}")
            st.write(f"⭐ 평점: {r['rating']} | 🚶 {r['distance']}분 | 💰 {r['price']}원")
            st.divider()

    # -------------------------------
    # 비교 차트
    # -------------------------------
    st.subheader("📊 식당 지표 비교")

    fig, ax = plt.subplots()
    ax.bar(df["name"], df["rating"])
    ax.set_ylabel("평점")
    ax.set_ylim(0, 5)

    st.pyplot(fig)

else:
    st.info("👆 상황을 입력하고 **점심 추천 받기** 버튼을 눌러주세요")
