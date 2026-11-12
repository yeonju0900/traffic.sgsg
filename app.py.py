import os
import pickle
import numpy as np
import streamlit as st
from Orange.data import Table

# ---------------------------------------------------------
# 1. 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="교통사고 사망자 예측 및 스쿨존 안전 신호등",
    page_icon="🚦",
    layout="wide",
)

MODEL_PATH = "traffic_casualty_model.pkcls"


# ---------------------------------------------------------
# 2. Orange3 모델 로드 (캐싱 적용)
# ---------------------------------------------------------
@st.cache_resource
def load_orange_model(model_path: str):
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


model = load_orange_model(MODEL_PATH)

# ---------------------------------------------------------
# 3. 헤더 영역
# ---------------------------------------------------------
st.title("🚦 교통사고 예상 사망자 수 예측 & 어린이 보호구역 안전 신호등")
st.markdown(
    "시도 선택 및 **사고건수·경상자수**를 슬라이더로 조절하면 Orange3 학습 모델이 **예상 사망자 수**를 실시간으로 계산하고, 위험도에 따른 **대형 신호등과 스쿨존 안전 수칙**을 안내합니다."
)
st.divider()

if model is None:
    st.error(
        f"⚠️ 모델 파일(`{MODEL_PATH}`)을 찾을 수 없습니다. "
        "GitHub 저장소의 `app.py`와 같은 폴더에 `traffic_casualty_model.pkcls` 파일을 업로드해주세요."
    )
    st.stop()

# 모델의 original_domain에서 시도 목록 추출 (기본값: ['경기', '부산', '서울'])
domain = model.original_domain
sido_var = domain["시도"]
sido_options = list(sido_var.values) if hasattr(sido_var, "values") else ["경기", "부산", "서울"]


# ---------------------------------------------------------
# 4. 실시간 예측 함수
# ---------------------------------------------------------
def predict_casualties(
    trained_model,
    sido: str,
    accidents: float,
    minor_injuries: float,
    severe_injuries: float = np.nan,
    reported_injuries: float = np.nan,
) -> float:
    """
    Orange3 모델의 original_domain 형식에 맞춰 Table을 생성하고 예측값을 반환합니다.
    입력되지 않은 변수(np.nan)는 모델 내부의 ReplaceUnknowns 전처리기가 학습 평균값으로 자동 보정합니다.
    """
    orig_domain = trained_model.original_domain
    row = []
    for attr in orig_domain.attributes:
        if attr.name == "시도":
            row.append(sido)
        elif attr.name == "사고건수":
            row.append(float(accidents))
        elif attr.name == "중상자수":
            row.append(float(severe_injuries))
        elif attr.name == "경상자수":
            row.append(float(minor_injuries))
        elif attr.name == "부상신고자수":
            row.append(float(reported_injuries))
        else:
            row.append(np.nan)

    input_table = Table.from_list(orig_domain, [row])
    pred = trained_model(input_table)
    # 회귀 모델 특성상 음수가 나올 경우 0.0으로 보정
    return max(0.0, float(pred[0]))


# ---------------------------------------------------------
# 5. 사이드바: 신호등 판정 기준 설정 (선택적 조정 가능)
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 신호등 위험도 기준 설정")
    st.caption("예상 사망자 수(명)에 따른 신호등 전환 임계값을 조정할 수 있습니다.")
    safe_limit = st.number_input("🟢 안전 최대 기준 (미만)", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
    caution_limit = st.number_input("🟡 주의 최대 기준 (미만)", min_value=2.0, max_value=100.0, value=20.0, step=1.0)
    if caution_limit <= safe_limit:
        st.warning("주의 기준은 안전 기준보다 커야 합니다.")
        caution_limit = safe_limit + 5.0

    st.divider()
    st.markdown("**📊 모델 정보**")
    st.write(f"- 알고리즘: `{getattr(model, 'name', 'Neural Network')}`")
    st.write(f"- 타깃 변수: `{domain.class_var.name}`")
    st.write(f"- 지원 시도: `{', '.join(sido_options)}`")


# ---------------------------------------------------------
# 6. 메인 레이아웃: 좌측(입력 컨트롤) / 우측(대형 신호등 & 결과)
# ---------------------------------------------------------
col_input, col_result = st.columns([1, 1.3], gap="large")

with col_input:
    st.subheader("🎛️ 교통사고 조건 입력")

    selected_sido = st.selectbox(
        "📍 시도 선택",
        options=sido_options,
        index=sido_options.index("서울") if "서울" in sido_options else 0,
    )

    accident_count = st.slider(
        "🚗 사고건수 (건)",
        min_value=10,
        max_value=3500,
        value=1000,
        step=10,
        help="슬라이더를 움직이면 즉시 사망자 수가 재계산됩니다.",
    )

    minor_injury_count = st.slider(
        "🩹 경상자수 (명)",
        min_value=10,
        max_value=4000,
        value=1200,
        step=10,
        help="슬라이더를 움직이면 즉시 사망자 수가 재계산됩니다.",
    )

    with st.expander("🔧 추가 변수 세부 설정 (선택 사항)"):
        st.caption("체크 해제 시 학습 데이터의 평균값(중상자 약 318명, 부상신고자 약 145명)이 자동 적용됩니다.")
        use_custom_extra = st.checkbox("중상자수·부상신고자수 직접 입력", value=False)
        if use_custom_extra:
            severe_count = st.slider("🚑 중상자수 (명)", min_value=0, max_value=1200, value=318, step=5)
            reported_count = st.slider("📋 부상신고자수 (명)", min_value=0, max_value=1000, value=145, step=5)
        else:
            severe_count = np.nan
            reported_count = np.nan

# 실시간 모델 예측 실행
predicted_deaths = predict_casualties(
    trained_model=model,
    sido=selected_sido,
    accidents=accident_count,
    minor_injuries=minor_injury_count,
    severe_injuries=severe_count,
    reported_injuries=reported_count,
)

# 예측값에 따른 신호등 상태 결정
if predicted_deaths < safe_limit:
    status_key = "SAFE"
    status_badge = "🟢 안전"
    status_title = "안전 단계 — 교통사고 사망 위험이 낮습니다"
    accent_color = "#16a34a"
    bg_tint = "#f0fdf4"
    border_color = "#86efac"
    red_style = "background:#3f1d1d; opacity:0.25;"
    yellow_style = "background:#3f3517; opacity:0.25;"
    green_style = "background:#22c55e; box-shadow: 0 0 32px #22c55e, 0 0 60px #16a34a;"
elif predicted_deaths < caution_limit:
    status_key = "CAUTION"
    status_badge = "🟡 주의"
    status_title = "주의 단계 — 감속 운행 및 집중 관리가 필요합니다"
    accent_color = "#d97706"
    bg_tint = "#fffbeb"
    border_color = "#fcd34d"
    red_style = "background:#3f1d1d; opacity:0.25;"
    yellow_style = "background:#facc15; box-shadow: 0 0 32px #facc15, 0 0 60px #eab308;"
    green_style = "background:#143522; opacity:0.25;"
else:
    status_key = "DANGER"
    status_badge = "🔴 위험"
    status_title = "위험 단계 — 중대 인명피해 우려! 즉각적인 안전 조치가 필요합니다"
    accent_color = "#dc2626"
    bg_tint = "#fef2f2"
    border_color = "#fca5a5"
    red_style = "background:#ef4444; box-shadow: 0 0 32px #ef4444, 0 0 60px #dc2626;"
    yellow_style = "background:#3f3517; opacity:0.25;"
    green_style = "background:#143522; opacity:0.25;"


with col_result:
    st.subheader("🚨 실시간 예측 결과 및 대형 신호등")

    traffic_light_html = f"""
    <div style="
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        background: {bg_tint};
        border: 2px solid {border_color};
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 12px;
    ">
        <!-- 대형 3구 신호등 본체 -->
        <div style="
            background: #1e293b;
            padding: 18px 20px;
            border-radius: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.25);
            border: 3px solid #334155;
            min-width: 108px;
            align-items: center;
        ">
            <div style="width: 64px; height: 64px; border-radius: 50%; {red_style} transition: all 0.25s;"></div>
            <div style="width: 64px; height: 64px; border-radius: 50%; {yellow_style} transition: all 0.25s;"></div>
            <div style="width: 64px; height: 64px; border-radius: 50%; {green_style} transition: all 0.25s;"></div>
        </div>

        <!-- 예측 수치 및 상태 텍스트 -->
        <div style="flex: 1;">
            <div style="font-size: 1.1rem; font-weight: 700; color: {accent_color}; margin-bottom: 6px;">
                [{selected_sido}] 실시간 위험도 판정
            </div>
            <div style="font-size: 2.6rem; font-weight: 900; color: {accent_color}; line-height: 1.15;">
                {status_badge}
            </div>
            <div style="font-size: 1.05rem; font-weight: 600; color: #1f2937; margin-top: 8px;">
                {status_title}
            </div>
            <div style="
                margin-top: 16px;
                padding: 12px 16px;
                background: #ffffff;
                border-radius: 12px;
                border: 1px solid {border_color};
            ">
                <span style="font-size: 0.95rem; color: #4b5563;">모델 예상 사망자 수: </span>
                <span style="font-size: 1.85rem; font-weight: 800; color: #111827;">
                    {predicted_deaths:.2f} 명
                </span>
                <span style="font-size: 0.85rem; color: #6b7280;">
                    (반올림 약 {round(predicted_deaths)}명)
                </span>
            </div>
        </div>
    </div>
    """
    st.markdown(traffic_light_html, unsafe_allow_html=True)


# ---------------------------------------------------------
# 7. 어린이 보호구역(스쿨존) 안전 수칙 카드 섹션
# ---------------------------------------------------------
st.divider()
st.subheader(f"🏫 어린이 보호구역(스쿨존) 맞춤형 안전 수칙 카드 — {status_badge}")

SAFETY_CARDS = {
    "SAFE": [
        {
            "icon": "🚗",
            "title": "운전자 기본 수칙 (시속 30km 준수)",
            "items": [
                "스쿨존 진입 시 시속 30km 이하 서행을 생활화하세요.",
                "신호등이 없는 횡단보도 앞에서도 반드시 일시정지 후 출발하세요.",
                "주정차 전면 금지 구역 내 불법 주정차를 절대 하지 마세요.",
            ],
        },
        {
            "icon": "🧒",
            "title": "어린이 보행자 안전 수칙",
            "items": [
                "횡단보도를 건너기 전 멈춰 서서 좌우 차량을 확인하세요.",
                "녹색불이 켜져도 차가 완전히 멈춘 것을 보고 건너세요.",
                "보행 중 스마트폰이나 이어폰 사용을 자제하세요.",
            ],
        },
        {
            "icon": "🛡️",
            "title": "지역 안전 예방 관리",
            "items": [
                "등·하교 시간대 노란색 횡단보도 및 시인성 표지판을 점검하세요.",
                "골목길·이면도로 교차 지점 반사경과 과속방지턱 상태를 유지하세요.",
                "어린이 통학버스 승·하차 시 보호자 동행 원칙을 준수하세요.",
            ],
        },
    ],
    "CAUTION": [
        {
            "icon": "⚠️",
            "title": "운전자 주의 수칙 (방어운전 강화)",
            "items": [
                "사고 빈도가 높은 구간입니다. 시속 20~30km 이하로 감속 운행하세요.",
                "주정차된 차량 사이에서 어린이가 갑자기 뛰어나올 수 있으니 전방을 주시하세요.",
                "우회전 시 횡단보도 보행자 유무를 확인하고 반드시 일시정지하세요.",
            ],
        },
        {
            "icon": "🚸",
            "title": "어린이 보행 집중 지도",
            "items": [
                "비·눈이 오거나 어두운 시간대에는 밝은 색 옷과 안전 우산을 착용하세요.",
                "차량 뒤편이나 사각지대에서 장난치거나 머무르지 마세요.",
                "육교나 지정된 횡단보도 외 무단횡단을 절대 금지합니다.",
            ],
        },
        {
            "icon": "📢",
            "title": "지자체·학교 주의 단계 대응",
            "items": [
                "등·하교 피크 시간대 녹색어머니회 및 교통안전지도사 배치를 확대하세요.",
                "스쿨존 내 불법 주정차 단속 카메라 및 순찰 빈도를 높이세요.",
                "학부모 및 운전자 대상 스쿨존 사고 다발 구간 알림을 발송하세요.",
            ],
        },
    ],
    "DANGER": [
        {
            "icon": "🚨",
            "title": "운전자 긴급 안전 수칙 (최고 수준 경계)",
            "items": [
                "사망사고 위험이 매우 높은 상태입니다. 스쿨존 내 서행(20km/h 이하) 및 즉각 정지 태세를 유지하세요.",
                "어린이 통학버스 추월을 절대 금지하고 비상등 점멸 시 정차하세요.",
                "교차로·횡단보도 인근 불법 주정차 차량은 즉시 이동 조치하세요.",
            ],
        },
        {
            "icon": "🛑",
            "title": "어린이 보호 최우선 행동 요령",
            "items": [
                "저학년 어린이는 가급적 보호자 또는 보행안전지도사와 함께 동행 하교하세요.",
                "횡단보도 대기 시 차도에서 한 걸음 이상 물러나 안전선 뒤에서 대기하세요.",
                "공이나 소지품이 차도로 굴러가도 절대 혼자 따라 뛰어들지 마세요.",
            ],
        },
        {
            "icon": "🚓",
            "title": "긴급 교통안전 대책 권고",
            "items": [
                "해당 시도·권역 스쿨존 내 과속·신호위반 집중 단속을 즉시 시행하세요.",
                "보행자 방호울타리, 바닥형 보행신호등, 대각선 횡단보도 등 안전시설을 긴급 보강하세요.",
                "사고 다발 시간대 경찰·모범운전자 합동 현장 통제를 실시하세요.",
            ],
        },
    ],
}

cards = SAFETY_CARDS[status_key]
card_cols = st.columns(3, gap="medium")

for col, card in zip(card_cols, cards):
    with col:
        items_html = "".join(
            f"<li style='margin-bottom: 8px; line-height: 1.5; color: #374151;'>{item}</li>"
            for item in card["items"]
        )
        card_html = f"""
        <div style="
            background: #ffffff;
            border: 1.5px solid {border_color};
            border-top: 6px solid {accent_color};
            border-radius: 14px;
            padding: 20px;
            height: 100%;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        ">
            <div style="font-size: 1.5rem; margin-bottom: 6px;">{card['icon']}</div>
            <div style="font-size: 1.08rem; font-weight: 800; color: #111827; margin-bottom: 12px;">
                {card['title']}
            </div>
            <ul style="padding-left: 18px; margin: 0; font-size: 0.94rem;">
                {items_html}
            </ul>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)