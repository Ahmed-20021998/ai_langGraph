import os
from typing import TypedDict, Literal

from dotenv import load_dotenv
import ollama
from openai import OpenAI

from langgraph.graph import StateGraph, START, END

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing")

if not OLLAMA_API_KEY:
    raise ValueError("OLLAMA_API_KEY is missing")


# ============================================================
# OLLAMA CLOUD CLIENT
# ============================================================

ollama_client = ollama.Client(
    host="https://ollama.com",
    headers={
        "Authorization": f"Bearer {OLLAMA_API_KEY}"
    }
)


# ============================================================
# OPENROUTER CLIENT
# ============================================================

openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)


# ============================================================
# STATE
# ============================================================

class State(TypedDict):

    user_message: str

    ollama_response: str

    claude_decision: str
    claude_response: str
    claude_additions: str
    claude_reasoning: str

    ollama_decision: str
    ollama_reasoning: str

    final_response: str


# ============================================================
# NODE 1 — OLLAMA CLOUD GENERATES FIRST RESPONSE
# ============================================================

def ollama_generate(state: State):

    print("\n[Ollama Cloud] Generating response...")

    user_message = state["user_message"]

    prompt = f"""
You are the first AI assistant.

Answer the user's question accurately and clearly.

If the question involves any calculation, numeric comparison,
or unit conversion, show your work step by step BEFORE giving
the final answer, then state the final answer clearly on its
own line at the end, prefixed with "Final answer:".

User question:
{user_message}
"""

    response = ollama_client.chat(
        model="gemma4:31b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response["message"]["content"]

    print("[Ollama Cloud] Response received.")

    return {
        "ollama_response": content
    }


# ============================================================
# NODE 2 — OPENROUTER REVIEWS OLLAMA
# ============================================================

def claude_review(state: State):

    print("\n[OpenRouter] Reviewing Ollama response...")

    user_message = state["user_message"]
    ollama_response = state["ollama_response"]

    prompt = f"""
You are a strict, careful reviewer.

You will be graded on ACCURACY, not on being agreeable or
on being different for the sake of it.

The user asked:

{user_message}

Ollama answered:

{ollama_response}

----------------------------------------------------------------
STEP 1 — INDEPENDENT VERIFICATION

Solve the user's question yourself from scratch.

If it involves numbers, comparisons, or units, work through
them carefully.

Then compare your independently derived answer to Ollama's
answer.

----------------------------------------------------------------
STEP 2 — VERDICT

You MUST choose exactly one:

DISAGREE
ADDITIONS

Choose DISAGREE if:

- Ollama's final answer is incorrect
- Ollama's conclusion is incorrect
- Ollama contains an important factual error
- Ollama contains an important logical error

After DISAGREE, provide the corrected answer.

Choose ADDITIONS otherwise.

If Ollama is correct and complete:

ADDITIONS:
NONE

If something important is missing:

ADDITIONS:
<additional information>

----------------------------------------------------------------
OUTPUT FORMAT

SCRATCHPAD:
<your verification>

VERDICT: <DISAGREE or ADDITIONS>

<content>
"""

    response = openrouter.chat.completions.create(
        model="openai/gpt-5-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=2000
    )

    result = response.choices[0].message.content.strip()

    scratchpad = ""
    verdict_block = result

    if "VERDICT:" in result:

        parts = result.split("VERDICT:", 1)

        scratchpad = (
            parts[0]
            .replace("SCRATCHPAD:", "")
            .strip()
        )

        verdict_block = parts[1].strip()

    lines = verdict_block.split("\n", 1)

    decision_line = lines[0].strip().upper()

    if "DISAGREE" in decision_line:
        decision = "DISAGREE"

    elif "ADDITIONS" in decision_line:
        decision = "ADDITIONS"

    else:
        decision = "DISAGREE"

    details = (
        lines[1].strip()
        if len(lines) > 1
        else ""
    )

    is_none_addition = (
        decision == "ADDITIONS"
        and (
            details == ""
            or details.strip().upper().rstrip(".") == "NONE"
        )
    )

    if decision == "DISAGREE":

        print("[OpenRouter Decision] DISAGREE")

    elif is_none_addition:

        print(
            "[OpenRouter Decision] "
            "ADDITIONS -> NONE"
        )

    else:

        print("[OpenRouter Decision] ADDITIONS")

    return {
        "claude_decision": decision,

        "claude_response": (
            details
            if decision == "DISAGREE"
            else ""
        ),

        "claude_additions": (
            ""
            if is_none_addition
            else (
                details
                if decision == "ADDITIONS"
                else ""
            )
        ),

        "claude_reasoning": scratchpad
    }


# ============================================================
# EDGE — CLAUDE DECISION
# ============================================================

def claude_decision(
    state: State
) -> Literal["agree", "disagree", "additions"]:

    decision = state["claude_decision"]

    additions = state["claude_additions"]

    if decision == "DISAGREE":
        return "disagree"

    if additions.strip() == "":
        return "agree"

    return "additions"


# ============================================================
# NODE 3 — RETURN OLLAMA
# ============================================================

def return_ollama(state: State):

    print(
        "[System] Returning Ollama's answer."
    )

    return {
        "final_response":
            state["ollama_response"]
    }


# ============================================================
# NODE 4 — PREPARE CLAUDE CORRECTION
# ============================================================

def prepare_claude_correction(state: State):

    return {
        "claude_response":
            state["claude_response"]
    }


# ============================================================
# NODE 5 — OLLAMA CLOUD REVIEWS CLAUDE
# ============================================================

def ollama_review_claude(state: State):

    print(
        "[Ollama Cloud] "
        "Reviewing OpenRouter's correction..."
    )

    user_message = state["user_message"]

    ollama_response = state["ollama_response"]

    claude_response_text = state["claude_response"]

    prompt = f"""
You are the second reviewer.

You will be graded on ACCURACY.

The user asked:

{user_message}

Your original answer was:

{ollama_response}

Another AI reviewer disagreed and proposed:

{claude_response_text}

----------------------------------------------------------------
STEP 1

Re-derive the answer yourself independently.

If the question involves numbers, comparisons, or units,
work through it step by step.

----------------------------------------------------------------
STEP 2

Compare your fresh result with the other reviewer's answer.

----------------------------------------------------------------
STEP 3

Output exactly one verdict:

AGREE

or

DISAGREE

----------------------------------------------------------------
OUTPUT FORMAT

SCRATCHPAD:
<your independent reasoning>

VERDICT: <AGREE or DISAGREE>
"""

    response = ollama_client.chat(
        model="gemma4:31b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw = response["message"]["content"].strip()

    scratchpad = ""

    verdict_text = raw

    if "VERDICT:" in raw:

        parts = raw.split("VERDICT:", 1)

        scratchpad = (
            parts[0]
            .replace("SCRATCHPAD:", "")
            .strip()
        )

        verdict_text = parts[1].strip().upper()

    else:

        verdict_text = raw.upper()

    if (
        "AGREE" in verdict_text
        and "DISAGREE" not in verdict_text
    ):
        decision = "AGREE"

    else:
        decision = "DISAGREE"

    print(
        f"[Ollama Decision] {decision}"
    )

    return {
        "ollama_decision": decision,
        "ollama_reasoning": scratchpad
    }


# ============================================================
# EDGE — OLLAMA DECISION
# ============================================================

def ollama_decision(
    state: State
) -> Literal["agree", "disagree"]:

    return (
        "agree"
        if state["ollama_decision"] == "AGREE"
        else "disagree"
    )


# ============================================================
# NODE 6 — RETURN CLAUDE CORRECTION
# ============================================================

def return_claude(state: State):

    print(
        "[System] "
        "Ollama agreed with OpenRouter correction."
    )

    return {
        "final_response":
            state["claude_response"]
    }


# ============================================================
# NODE 7 — RETURN OLLAMA + ADDITIONS
# ============================================================

def return_ollama_with_additions(
    state: State
):

    print(
        "[System] "
        "Returning Ollama + OpenRouter additions."
    )

    return {
        "final_response":
            state["ollama_response"]
            + "\n\n"
            + state["claude_additions"]
    }


# ============================================================
# BUILD GRAPH
# ============================================================

builder = StateGraph(State)

builder.add_node(
    "ollama_generate",
    ollama_generate
)

builder.add_node(
    "claude_review",
    claude_review
)

builder.add_node(
    "return_ollama",
    return_ollama
)

builder.add_node(
    "prepare_claude_correction",
    prepare_claude_correction
)

builder.add_node(
    "ollama_review_claude",
    ollama_review_claude
)

builder.add_node(
    "return_claude",
    return_claude
)

builder.add_node(
    "return_ollama_with_additions",
    return_ollama_with_additions
)


# ============================================================
# GRAPH EDGES
# ============================================================

builder.add_edge(
    START,
    "ollama_generate"
)

builder.add_edge(
    "ollama_generate",
    "claude_review"
)


builder.add_conditional_edges(
    "claude_review",
    claude_decision,
    {
        "agree":
            "return_ollama",

        "disagree":
            "prepare_claude_correction",

        "additions":
            "return_ollama_with_additions"
    }
)


builder.add_edge(
    "prepare_claude_correction",
    "ollama_review_claude"
)


builder.add_conditional_edges(
    "ollama_review_claude",
    ollama_decision,
    {
        "agree":
            "return_claude",

        "disagree":
            "return_ollama"
    }
)


builder.add_edge(
    "return_ollama",
    END
)

builder.add_edge(
    "return_claude",
    END
)

builder.add_edge(
    "return_ollama_with_additions",
    END
)


app_graph = builder.compile()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Dual-AI Review API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-lang-graph-jj5z-kw1iyeghs-ahmeds-projects-5024b40d.vercel.app/ask",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):

    message: str


class AskResponse(BaseModel):

    final_response: str

    claude_decision: str

    ollama_decision: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def health():

    return {
        "status": "ok"
    }


# ============================================================
# ASK
# ============================================================

@app.post(
    "/ask",
    response_model=AskResponse
)
def ask(payload: AskRequest):

    if (
        not payload.message
        or not payload.message.strip()
    ):
        raise HTTPException(
            status_code=400,
            detail="message must not be empty"
        )

    print(
        f"\n[Request] {payload.message}"
    )

    try:

        result = app_graph.invoke({

            "user_message":
                payload.message,

            "ollama_response":
                "",

            "claude_decision":
                "",

            "claude_response":
                "",

            "claude_additions":
                "",

            "claude_reasoning":
                "",

            "ollama_decision":
                "",

            "ollama_reasoning":
                "",

            "final_response":
                ""
        })

    except Exception as e:

        print(
            f"[Pipeline Error] {repr(e)}"
        )

        raise HTTPException(
            status_code=502,
            detail=f"Pipeline error: {e}"
        )

    return AskResponse(

        final_response=
            result["final_response"],

        claude_decision=
            result.get(
                "claude_decision",
                ""
            ),

        ollama_decision=
            result.get(
                "ollama_decision",
                ""
            )
    )
