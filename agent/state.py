# agent/state.py
# ─────────────────────────────────────────────
# Yeh file agent ki "memory" hai.
# Agent ko yaad rakhna padta hai:
#   - User ne kya kaha
#   - Data ka kya scene hai
#   - Kya analysis hua
# ─────────────────────────────────────────────

from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """Agent ki memory — har step ke baad update hoti hai."""

    # 1) Chat history — user aur agent ke beech poori baat-cheet
    #    "operator.add" ka matlab: naye messages puranon mein ADD hote hain
    #    (replace nahi hote — history banti rehti hai)
    messages: Annotated[list[BaseMessage], operator.add]

    # 2) CSV file ka path — user ne kaunsi file di
    file_path: str

    # 3) Data ka summary — columns, types, pehli 5 rows
    data_summary: str

    # 4) Analysis ka result — pandas ne jo nikala
    analysis_result: str

    # 5) Chart ka path — agar graph bana toh kahan save hua
    chart_path: str
