

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    # python-dotenv is not installed in the environment (e.g., deployment or CI).
    # Provide a no-op fallback so the rest of the app can use environment variables
    # from the operating system without raising an ImportError.
    def load_dotenv():
        return

load_dotenv()
import os
# Install dependencies in your environment (do NOT run pip via an import statement inside the script).
# Example (run in your shell/venv): pip install streamlit==1.38.0 langchain==0.3.26 langchain-openai==0.2.7 openai==1.54.0
import streamlit as st

# Try multiple possible LangChain import paths to avoid editor/installation import errors.
# Preferred: langchain.chat_models.ChatOpenAI (newer LangChain).
# Fallback: langchain.llms.OpenAI (older/different packaging) aliased to ChatOpenAI.
try:
    from langchain.chat_models import ChatOpenAI
except Exception:
    try:
        from langchain.llms import OpenAI as ChatOpenAI
    except Exception:
        ChatOpenAI = None  # Will be checked at runtime and a helpful error will be raised.

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# ========== 設定 ==========
MODEL_NAME = "gpt-4o-mini"  # 必要に応じて変更
TEMPERATURE = 0.3

# A/B の専門家プロンプトをここで定義（増やすのも簡単：辞書に追加）
EXPERT_SYSTEM_MESSAGES = {
    "A": (
        "あなたは『A: ビジネス戦略コンサルタント』です。"
        "ユーザーの入力テキストを精読し、実務で使える打ち手・優先順位・理由を"
        "簡潔かつ具体例つきで提案してください。"
        "制約: 箇条書き中心、最初に要約（3行以内）、指標/KPI案があれば数値例も。"
    ),
    "B": (
        "あなたは『B: 学習デザイン/教育コーチ』です。"
        "学習者中心の観点から、学習目標、学習ステップ、指導法、評価方法を"
        "具体的に設計して提案してください。"
        "制約: 実行可能なステップと評価指標（KPI）を明示し、例を挙げてください。"
    ),
}

def build_chain(system_message: str):
    """Systemメッセージを受け取り、LangChainの実行チェーンを構築して返す。"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", "{user_input}")
        ]
    )

    if ChatOpenAI is None:
        raise RuntimeError(
            "LangChain ChatOpenAI is not available; please install a compatible langchain package "
            "(e.g. `pip install langchain`) and ensure your environment has the correct version."
        )

    # Different LangChain versions may expose different classes/parameter names.
    # Try the common constructor first, then fallback to an alternative parameter name.
    try:
        llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
    except TypeError:
        llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain

def get_llm_response(user_text: str, expert_key: str) -> str:
    """
    要件：『入力テキスト』『ラジオボタン選択値』を受け取り、
    LLMからの回答を戻り値として返す関数。

    Parameters
    ----------
    user_text : str
        入力フォームでユーザーが入力したテキスト
    expert_key : str
        ラジオボタンで選んだ専門家キー（例: "A", "B"）

    Returns
    -------
    str
        モデルの回答テキスト
    """
    system_message = EXPERT_SYSTEM_MESSAGES.get(
        expert_key,
        EXPERT_SYSTEM_MESSAGES["A"]  # フォールバック
    )
    chain = build_chain(system_message)
    return chain.invoke({"user_input": user_text})


# ========== Streamlit UI ==========
st.set_page_config(page_title="LLMヘルプデスク（LangChain）", page_icon="🤖", layout="centered")

st.title("🤖 LLMヘルプデスク（LangChain x Streamlit）")
st.caption("入力テキストを専門家ペルソナ（A/B）で解釈し、最適な回答を返します。")

with st.expander("ℹ️ アプリ概要・使い方", expanded=True):
    st.markdown(
        """
**このアプリでできること**
- 画面のテキスト入力を、そのままLLMへのプロンプトとして送信します  
- ラジオボタンで専門家の役割（A/B）を切り替え、**システムメッセージ**を変更して応答の観点を調整します  
- LangChain（PromptTemplate → ChatOpenAI → OutputParser）で構成

**操作方法**
1. 下のラジオボタンで専門家タイプ（A または B）を選びます  
2. テキストエリアに相談内容・要件・質問などを入力します  
3. **送信**ボタンを押すと、画面下部に回答が表示されます

**ヒント**
- 回答の粒度やトーンを変えたい場合は、A/Bを切り替えて比較してみてください  
- 専門家タイプは `EXPERT_SYSTEM_MESSAGES` に追記すれば簡単に増やせます
        """
    )

# --- 入力UI ---
expert_choice_label = st.radio(
    "専門家の種類を選択してください：",
    options=["A", "B"],
    format_func=lambda k: f"{k}: " + (
        "ビジネス戦略コンサル" if k == "A" else
        "学習デザイン/教育コーチ"
    ),
    horizontal=True
)

user_text = st.text_area(
    "入力テキスト",
    placeholder="例）新サービスの初期集客施策を90日で設計したい。少人数・低予算で実現するプランとKPIを…",
    height=180
)

submit = st.button("送信", type="primary")

# --- 実行＆表示 ---
if submit:
    if not user_text.strip():
        st.warning("テキストを入力してください。")
    else:
        with st.spinner("LLMが回答を作成中..."):
            try:
                answer = get_llm_response(user_text, expert_choice_label)
                st.markdown("### 🧠 回答")
                st.write(answer)
            except Exception as e:
                st.error("エラーが発生しました。設定やAPIキーをご確認ください。")
                with st.expander("エラー詳細"):
                    st.exception(e)

# フッター
st.markdown("---")
st.caption(
    "Powered by LangChain + OpenAI | System prompts switch with radio selection (A/B)."
)
