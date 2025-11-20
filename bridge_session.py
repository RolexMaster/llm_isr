from __future__ import annotations

# =========================
# 패키지 버전 / 커밋 정보 (pip 설치 버전 확인용)
# =========================
# 릴리스할 때마다 __version__ 값을 pyproject.toml / setup.cfg 의 버전과
# 반드시 동일하게 맞춰주세요.
__version__ = "0.1.0"
__commit__ = "dev"  # 필요하면 실제 git short SHA 등으로 교체해서 사용

import json
import time
import re
import logging
from typing import Dict, Any, List, Tuple

from openai import OpenAI
from fastmcp import Client

from agent_prompts import build_system_prompt, AgentLanguage  # 🔹 새로 추가


# =========================
# 로컬 유틸
# =========================
def pretty_json(obj: Any, limit: int = 1000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + "\n... (truncated)"


# (현재는 정규식 블록은 사용하지 않지만, 필요시를 위해 남겨둠)
TOOLCALL_BLOCK = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.DOTALL
)


def extract_toolcalls_from_text(text: str) -> List[Dict[str, Any]]:
    """
    모델이 tool_calls를 구조화해 주지 않고,
    텍스트로 <tool_call>{ "name": ..., "arguments": {...} }</tool_call> 를 낸 경우 파싱.
    정규식 대신 중괄호 밸런싱으로 첫 번째 JSON 블록을 최대한 관대하게 추출한다.
    - <tool_call>{...}</tool_call> 형태는 물론
    - <tool_call>{...}<tool_call> 처럼 닫는 태그가 빠져 있어도
      첫 번째 {...} 블록만 잘라서 파싱을 시도한다.
    """
    out: List[Dict[str, Any]] = []
    if not text:
        return out

    # '<tool_call>' 기준으로 잘라서 각 블록에서 JSON 부분을 찾는다.
    parts = text.split("<tool_call>")
    if len(parts) <= 1:
        return out

    for part in parts[1:]:
        # part 예:
        #   '{"name": "...", "arguments": {...}}</tool_call> bla bla'
        #   '{"name": "...", "arguments": {...}}<tool_call>{...}</tool_call>...'
        # 에서 첫 번째 {...}만 추출
        brace_start = part.find("{")
        if brace_start < 0:
            continue

        depth = 0
        brace_end = None
        for idx in range(brace_start, len(part)):
            ch = part[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    brace_end = idx
                    break

        if brace_end is None:
            # 중괄호 균형이 맞지 않으면 스킵
            continue

        json_str = part[brace_start: brace_end + 1]

        try:
            obj = json.loads(json_str)
        except Exception:
            # JSON 파싱 실패 시 스킵
            continue

        name = obj.get("name")
        args = obj.get("arguments", {}) or {}
        if isinstance(name, str) and isinstance(args, dict):
            out.append({"name": name, "arguments": args})

    return out


def requires_params_wrapper(schema_json: dict) -> bool:
    """서버가 최상위 {"params": {...}} 형태를 요구하는지 입력 스키마로 추정."""
    if not isinstance(schema_json, dict):
        return False
    props = schema_json.get("properties", {})
    req = set(schema_json.get("required", []))
    return ("params" in props) and ("params" in req)


def build_name_maps(mcp_tools: List[Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """llm_func_name <-> mcp_name 매핑 생성"""
    f2m: Dict[str, str] = {}
    m2f: Dict[str, str] = {}
    for t in mcp_tools:
        name = getattr(t, "name", None) or t.get("name")
        if not name:
            continue
        llm_name = name.replace(".", "_")
        f2m[llm_name] = name
        m2f[name] = llm_name
    return f2m, m2f


def build_tools_block(mcp_tools: List[Any]) -> str:
    """
    시스템 프롬프트에 주입할 간단한 도구 목록(JSON).
    name과 parameters만 싣는다.
    """
    items: List[Dict[str, Any]] = []
    for t in mcp_tools:
        name = getattr(t, "name", None) or t.get("name")
        if not name:
            continue

        schema = (
            getattr(t, "input_schema", None)
            or getattr(t, "inputSchema", None)
            or t.get("input_schema")
            or t.get("inputSchema")
            or {"type": "object", "properties": {}}
        )
        if hasattr(schema, "model_dump"):
            schema = schema.model_dump()
        if not isinstance(schema, dict):
            schema = {"raw": str(schema)}

        items.append(
            {
                "name": name.replace(".", "_"),
                "parameters": schema,
            }
        )
    return json.dumps(items, ensure_ascii=False, indent=2)


# =========================
# BridgeSession
# =========================
class BridgeSession:
    """
    - LLM, MCP 클라이언트, MCP 도구목록을 가진 대화 세션 컨테이너
    - 시스템 프롬프트 구성/히스토리 유지
    - 사용자 입력 1건을 처리(handle_one_turn): 필요 시 MCP 도구를 호출하고, 최종 LLM 출력 문자열을 반환
    - debug=True 일 때는 LLM 각 턴/툴 호출까지 모두 담은 dict 반환
    """

    # 패키지 버전/커밋 정보 노출 (디버깅/버전 확인용)
    VERSION: str = __version__
    COMMIT: str = __commit__

    def __init__(
        self,
        llm: OpenAI,
        mcp: Client,
        mcp_tools: List[Any],
        *,
        temp: float = 0.2,
        max_tokens: int = 512,
        max_turns: int = 3,
        language: AgentLanguage = "en",  # 🔹 Llama 기본 영어 에이전트
    ):
        self.llm = llm
        self.mcp = mcp
        self.mcp_tools = mcp_tools
        self.temp = temp
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.language = language
        self.logger = logging.getLogger("mcp_bridge")

        # LLM 함수명 <-> MCP 도구명 매핑
        self.f2m, _ = build_name_maps(mcp_tools)
        self.tools_block = build_tools_block(mcp_tools)

        # 🔹 언어에 맞는 시스템 프롬프트 생성
        self.sys_prompt = build_system_prompt(
            language=self.language,
            tools_block=self.tools_block,
        )

        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.sys_prompt}
        ]

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.sys_prompt}]
        self.logger.info("[SESSION] messages reset")

    async def handle_one_turn(
        self,
        user_text: str,
        *,
        model: str,
        debug: bool = False,
    ) -> Any:
        """
        - 사용자 입력 1건 처리(필요하면 MCP 도구 호출 포함)
        - debug=False (기본값): 최종 LLM 출력 문자열만 반환
        - debug=True: LLM 각 턴의 raw 출력, tool_calls, tool_results까지 모두 담은 dict 반환
        """
        self.messages.append({"role": "user", "content": user_text})
        self.logger.debug(
            "[LLM INPUT FULL]\n" + pretty_json(self.messages, limit=100000)
        )

        assistant_text: str = ""
        turns_debug: List[Dict[str, Any]] = []

        # 도구 호출이 있을 수 있으므로 최대 self.max_turns 반복
        for turn in range(1, self.max_turns + 1):
            t0 = time.perf_counter()
            resp = self.llm.chat.completions.create(
                model=model,
                messages=self.messages,
                tool_choice="none",  # vLLM 내장 파서 비활성화 (폴백 파서 사용)
                temperature=self.temp,
                max_tokens=self.max_tokens,
            )
            dt = time.perf_counter() - t0

            msg = resp.choices[0].message
            self.logger.info(
                f"[LLM] response id: {getattr(resp, 'id', '<no-id>')} ({dt:.2f}s), "
                f"finish_reason={resp.choices[0].finish_reason}"
            )
            self.logger.debug(f"[LLM] assistant content:\n{msg.content}")

            assistant_text = (msg.content or "").strip()

            # 폴백 파서로 tool_call 추출
            calls: List[Dict[str, Any]] = []
            if msg.content and "<tool_call>" in msg.content:
                calls = extract_toolcalls_from_text(msg.content)
                self.logger.info(
                    f"[LLM] fallback extracted tool_calls: {calls}"
                )

            # 이번 턴에 대한 디버그 정보 틀
            turn_info: Dict[str, Any] = {
                "turn_index": turn,
                "llm_raw": assistant_text,  # LLM이 그대로 낸 텍스트
                "tool_calls": calls,        # 파싱된 도구 호출들
                "tool_results": [],         # 아래에서 채움
            }

            # 도구 호출이 없다면 최종 응답으로 종료
            if not calls:
                # 참고: 여기서 <tool_response>가 섞여 있으면 경고만 남기고 그대로 반환
                if "<tool_response>" in assistant_text:
                    self.logger.warning(
                        "[BRIDGE] LLM emitted forbidden <tool_response> tag in a non-tool_call turn."
                    )
                turns_debug.append(turn_info)
                if debug:
                    return {
                        "final_answer": assistant_text,
                        "turns": turns_debug,
                    }
                return assistant_text

            # 도구 호출 실행
            all_ok = True
            tool_results_turn: List[Dict[str, Any]] = []

            for i, c in enumerate(calls, 1):
                func_name = c["name"]
                args = c.get("arguments", {})
                self.logger.info(
                    f"[TOOLCALL {i}] func={func_name} args={json.dumps(args, ensure_ascii=False)}"
                )
                mcp_name = self.f2m.get(func_name)
                self.logger.info(
                    f"[TOOLCALL {i}] mapped LLM '{func_name}' -> MCP '{mcp_name}'"
                )

                # 기본 payload는 그냥 args (mcp_name이 없을 때도 사용 가능하도록)
                payload: Dict[str, Any] = args
                tool_result: Dict[str, Any]

                if not mcp_name:
                    tool_result = {
                        "ok": False,
                        "error": f"unknown_tool_mapping:{func_name}",
                    }
                    all_ok = False
                    self.logger.error(
                        f"[TOOLCALL {i}] ERROR {tool_result['error']}"
                    )
                else:
                    # 필요시 {"params": {...}} 래핑
                    schema_obj = next(
                        (
                            getattr(t, "input_schema", None)
                            or getattr(t, "inputSchema", None)
                            or t.get("input_schema")
                            or t.get("inputSchema")
                            for t in self.mcp_tools
                            if (getattr(t, "name", None) or t.get("name")) == mcp_name
                        ),
                        None,
                    )

                    if hasattr(schema_obj, "model_dump"):
                        schema_json = schema_obj.model_dump()
                    elif isinstance(schema_obj, dict):
                        schema_json = schema_obj
                    else:
                        schema_json = {}

                    payload = (
                        {"params": args}
                        if requires_params_wrapper(schema_json)
                        else args
                    )

                    try:
                        self.logger.info(
                            f"[TOOLCALL {i}] calling MCP tools/call name={mcp_name} payload={payload}"
                        )
                        t1 = time.perf_counter()
                        res = await self.mcp.call_tool(mcp_name, payload)
                        dtt = time.perf_counter() - t1
                        tool_result = getattr(res, "data", res)
                        self.logger.info(
                            f"[TOOLCALL {i}] MCP result ({dtt:.2f}s):\n{pretty_json(tool_result)}"
                        )
                        # tool_result 형식 방어: dict 이면서 ok=True 일 때만 성공으로 간주
                        if not (isinstance(tool_result, dict) and tool_result.get("ok", False)):
                            all_ok = False
                    except Exception as e:
                        tool_result = {"ok": False, "error": str(e)}
                        all_ok = False
                        self.logger.exception(
                            f"[TOOLCALL {i}] MCP ERROR: {e}"
                        )

                # 이번 tool 호출에 대한 디버그 정보
                tool_results_turn.append(
                    {
                        "index": i,
                        "func_name": func_name,
                        "mcp_name": mcp_name,
                        "args": args,
                        "payload": payload,
                        "result": tool_result,
                    }
                )

                # tool 결과를 role=tool로 LLM에 피드백
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": f"tc_{i}",
                        "name": func_name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )
                self.logger.info(f"[TOOLCALL {i}] appended tool message")

            # 이번 턴의 tool 결과 채우고, 턴 디버그 리스트에 추가
            turn_info["tool_results"] = tool_results_turn
            turns_debug.append(turn_info)

            # ✅ 모든 도구 호출이 성공한 경우 → 바로 한 번 더 불러서 자연어 요약 후 종료
            if all_ok:
                self.logger.info(
                    "[BRIDGE] All tool calls OK; requesting final natural-language answer from LLM."
                )
                t2 = time.perf_counter()
                resp2 = self.llm.chat.completions.create(
                    model=model,
                    messages=self.messages,      # 여기에는 이미 role=tool 메시지들이 포함되어 있음
                    tool_choice="none",
                    temperature=self.temp,
                    max_tokens=self.max_tokens,
                )
                dt2 = time.perf_counter() - t2

                msg2 = resp2.choices[0].message
                self.logger.info(
                    f"[LLM FINAL] response id: {getattr(resp2, 'id', '<no-id>')} ({dt2:.2f}s), "
                    f"finish_reason={resp2.choices[0].finish_reason}"
                )
                self.logger.debug(f"[LLM FINAL] assistant content:\n{msg2.content}")

                final_text = (msg2.content or "").strip()

                if debug:
                    return {
                        "final_answer": final_text,
                        "turns": turns_debug,
                    }
                return final_text

            # ❌ 도구 중 일부 실패 → 다음 턴에서 다시 계획 짜도록 한 번 더 LLM 호출
            self.logger.info(
                "[BRIDGE] Some tool calls failed; asking LLM for another plan."
            )
            continue

        # 여기까지 왔다는 것은 self.max_turns 번 돌 때까지
        # '도구 호출 없는 최종 답변'을 못 받았다는 뜻
        self.logger.warning(
            "[CLIENT] Max turns reached without final non-tool answer; returning last assistant_text."
        )
        if debug:
            return {
                "final_answer": assistant_text,
                "turns": turns_debug,
            }
        return assistant_text
