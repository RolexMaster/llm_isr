# bridge_session.py
from __future__ import annotations
import json, time, re, logging
from typing import Dict, Any, List, Tuple, Optional
from openai import OpenAI
from fastmcp import Client

# =========================
# 로컬 유틸
# =========================
def pretty_json(obj: Any, limit: int = 1000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + "\n... (truncated)"

TOOLCALL_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.DOTALL)

def extract_toolcalls_from_text(text: str) -> List[Dict[str, Any]]:
    """
    모델이 tool_calls를 구조화해 주지 않고,
    텍스트로 <tool_call>{ "name": ..., "arguments": {...} }</tool_call> 를 낸 경우 파싱.
    """
    out: List[Dict[str, Any]] = []
    if not text:
        return out
    for m in TOOLCALL_BLOCK.finditer(text):
        try:
            obj = json.loads(m.group(1))
            name = obj.get("name")
            args = obj.get("arguments", {}) or {}
            if isinstance(name, str) and isinstance(args, dict):
                out.append({"name": name, "arguments": args})
        except Exception:
            pass
    return out

def requires_params_wrapper(schema_json: dict) -> bool:
    """ 서버가 최상위 {"params": {...}} 형태를 요구하는지 입력 스키마로 추정. """
    if not isinstance(schema_json, dict):
        return False
    props = schema_json.get("properties", {})
    req = set(schema_json.get("required", []))
    return ("params" in props) and ("params" in req)

def build_name_maps(mcp_tools: List[Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """ llm_func_name <-> mcp_name 매핑 생성 """
    f2m, m2f = {}, {}
    for t in mcp_tools:
        name = getattr(t, "name", None) or t.get("name")
        llm_name = name.replace(".", "_")
        f2m[llm_name] = name
        m2f[name] = llm_name
    return f2m, m2f

def build_tools_block(mcp_tools: List[Any]) -> str:
    """
    시스템 프롬프트에 주입할 간단한 도구 목록(JSON).
    name과 parameters만 싣는다.
    """
    items = []
    for t in mcp_tools:
        name = getattr(t, "name", None) or t.get("name")
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
        items.append({"name": name.replace(".", "_"), "parameters": schema})
    return json.dumps(items, ensure_ascii=False, indent=2)

# =========================
# BridgeSession
# =========================
class BridgeSession:
    """
    - LLM, MCP 클라이언트, MCP 도구목록을 가진 대화 세션 컨테이너
    - 시스템 프롬프트 구성/히스토리 유지
    - 사용자 입력 1건을 처리(handle_one_turn): 필요 시 MCP 도구를 호출하고, 최종 LLM 출력 문자열을 반환
    - 모든 내부 상세는 파일 로그(logger "mcp_bridge")로만 남김 (콘솔 출력 없음)
    """
    def __init__(
        self,
        llm: OpenAI,
        mcp: Client,
        mcp_tools: List[Any],
        *,
        temp: float = 0.2,
        max_tokens: int = 512,
        max_turns: int = 8,
        force_korean: bool = True,
    ):
        self.llm = llm
        self.mcp = mcp
        self.mcp_tools = mcp_tools
        self.temp = temp
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.logger = logging.getLogger("mcp_bridge")

        self.f2m, _ = build_name_maps(mcp_tools)
        self.tools_block = build_tools_block(mcp_tools)

        lang_rule = (
            "⚠️ 중요한 규칙: 모든 답변은 반드시 한국어로만 하십시오. 영어·중국어·기타 언어를 사용하지 마십시오.\n"
            "⚠️ 도구 호출은 반드시 <tool_call>...</tool_call> 형식만 사용하십시오. "
            "태그 없는 JSON이나 자연어 설명으로 도구를 호출하지 마십시오.\n"
            "⚠️ 도구 응답(role=tool)의 원시 JSON이나 로그를 사용자에게 절대 그대로 보여주지 마십시오. "
            "<tool_response>와 같은 임의 태그를 사용하지 마십시오. "
            "반드시 한국어 한두 문장으로 요약해 답하십시오.\n\n"
        ) if force_korean else (
            "⚠️ Do NOT echo raw tool JSON or logs to the user. "
            "NEVER use <tool_response>; summarize tool outcomes in natural language.\n\n"
        )
        
        self.sys_prompt = (
            f"{lang_rule}"
            "You are an AI assistant that can call external MCP tools.\n\n"
            "⚠️ Important formatting rules:\n"
            "1. When you need tools, emit one or more blocks in the following exact format:\n"
            "<tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>\n"
            "2. Always include both the opening <tool_call> and closing </tool_call> tags.\n"
            "3. Do not add extra text between <tool_call> blocks.\n"
            "4. Do not include natural language explanation inside tool_call blocks.\n"
            "5. Always use valid JSON inside the block.\n\n"
            "✅ Correct Example:\n"
            "<tool_call>{\"name\": \"eots_set_mode\", \"arguments\": {\"mode\": \"ir\"}}</tool_call>\n"
            "<tool_call>{\"name\": \"eots_pan_tilt\", \"arguments\": {\"pan_deg\": -20, \"tilt_deg\": 5}}</tool_call>\n"
            "<tool_call>{\"name\": \"eots_zoom\", \"arguments\": {\"level\": 8}}</tool_call>\n\n"
            "Available tools are listed as JSON schemas below (name and parameters only):\n"
            "{{TOOLS_BLOCK}}"
        ).replace("{{TOOLS_BLOCK}}", self.tools_block)
        
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": self.sys_prompt}]

    def reset(self):
        self.messages = [{"role": "system", "content": self.sys_prompt}]
        self.logger.info("[SESSION] messages reset")

    async def handle_one_turn(self, user_text: str, *, model: str) -> str:
        """
        - 사용자 입력 1건 처리(필요하면 MCP 도구 호출 포함)
        - LLM의 최종 출력 문자열을 반환
        """
        self.messages.append({"role": "user", "content": user_text})
        self.logger.debug("[LLM INPUT FULL]\n" + pretty_json(self.messages, limit=100000))

        assistant_text: str = ""
        # 도구 호출이 있을 수 있으므로 최대 self.max_turns 반복
        for turn in range(1, self.max_turns + 1):
            t0 = time.perf_counter()
            resp = self.llm.chat.completions.create(
                model=model,
                messages=self.messages,
                tool_choice="none",      # vLLM 내장 파서 비활성화 (폴백 파서 사용)
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
            calls = []
            if msg.content and "<tool_call>" in msg.content:
                calls = extract_toolcalls_from_text(msg.content)
                self.logger.info(f"[LLM] fallback extracted tool_calls: {calls}")

            # 도구 호출이 없다면 최종 응답으로 종료
            if not calls:
                return assistant_text

            # 도구 호출 실행
            all_ok = True
            for i, c in enumerate(calls, 1):
                func_name = c["name"]
                args = c.get("arguments", {})
                self.logger.info(f"[TOOLCALL {i}] func={func_name} args={json.dumps(args, ensure_ascii=False)}")
                mcp_name = self.f2m.get(func_name)
                self.logger.info(f"[TOOLCALL {i}] mapped LLM '{func_name}' -> MCP '{mcp_name}'")
                if not mcp_name:
                    tool_result = {"ok": False, "error": f"unknown_tool_mapping:{func_name}"}
                    all_ok = False
                    self.logger.error(f"[TOOLCALL {i}] ERROR {tool_result['error']}")
                else:
                    # 필요시 {"params": {...}} 래핑
                    schema_obj = next(
                        (getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)
                         or t.get("input_schema") or t.get("inputSchema")
                         for t in self.mcp_tools
                         if (getattr(t, "name", None) or t.get("name")) == mcp_name),
                        None
                    )
                    if hasattr(schema_obj, "model_dump"):
                        schema_json = schema_obj.model_dump()
                    elif isinstance(schema_obj, dict):
                        schema_json = schema_obj
                    else:
                        schema_json = {}
                    payload = {"params": args} if requires_params_wrapper(schema_json) else args
                    try:
                        self.logger.info(f"[TOOLCALL {i}] calling MCP tools/call name={mcp_name} payload={payload}")
                        t1 = time.perf_counter()
                        res = await self.mcp.call_tool(mcp_name, payload)
                        dtt = time.perf_counter() - t1
                        tool_result = getattr(res, "data", res)
                        self.logger.info(f"[TOOLCALL {i}] MCP result ({dtt:.2f}s):\n{pretty_json(tool_result)}")
                        if not tool_result or not tool_result.get("ok", False):
                            all_ok = False
                    except Exception as e:
                        tool_result = {"ok": False, "error": str(e)}
                        all_ok = False
                        self.logger.exception(f"[TOOLCALL {i}] MCP ERROR: {e}")

                # tool 결과를 role=tool로 LLM에 피드백
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": f"tc_{i}",
                    "name": func_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })
                self.logger.info(f"[TOOLCALL {i}] appended tool message")

            # 모든 MCP 호출 성공 → 후속 요약/확인 답변 한 턴 더 유도
            if all_ok:
                self.logger.info("[BRIDGE] All tool calls OK; requesting follow-up answer from LLM.")
                continue
            else:
                self.logger.info("[BRIDGE] Some tool calls failed; still asking LLM for follow-up.")
                continue

        self.logger.warning("[CLIENT] Max turns reached without final answer.")
        return assistant_text
