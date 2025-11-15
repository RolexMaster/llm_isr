# bridge_session.py (일부만 발췌)
from __future__ import annotations
import json, time, re, logging
from typing import Dict, Any, List, Tuple, Optional
from openai import OpenAI
from fastmcp import Client

# ... (위 유틸/클래스 정의들은 그대로 두고)

class BridgeSession:
    # __init__, reset 그대로 두고 아래 handle_one_turn만 교체
    async def handle_one_turn(self, user_text: str, *, model: str, debug: bool = False) -> Any:
        """
        - 사용자 입력 1건 처리(필요하면 MCP 도구 호출 포함)
        - debug=False (기본값): 최종 LLM 출력 문자열만 반환
        - debug=True: LLM 각 턴의 raw 출력, tool_calls, tool_results까지 모두 담은 dict 반환
        """
        self.messages.append({"role": "user", "content": user_text})
        self.logger.debug("[LLM INPUT FULL]\n" + pretty_json(self.messages, limit=100000))

        assistant_text: str = ""
        turns_debug: List[Dict[str, Any]] = []

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
            calls: List[Dict[str, Any]] = []
            if msg.content and "<tool_call>" in msg.content:
                calls = extract_toolcalls_from_text(msg.content)
                self.logger.info(f"[LLM] fallback extracted tool_calls: {calls}")

            # 이번 턴에 대한 디버그 정보 틀
            turn_info: Dict[str, Any] = {
                "turn_index": turn,
                "llm_raw": assistant_text,   # LLM이 그대로 낸 텍스트
                "tool_calls": calls,         # 파싱된 도구 호출들
                "tool_results": [],          # 아래에서 채움
            }

            # 도구 호출이 없다면 최종 응답으로 종료
            if not calls:
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

                # 이번 tool 호출에 대한 디버그 정보
                tool_results_turn.append({
                    "index": i,
                    "func_name": func_name,
                    "mcp_name": mcp_name,
                    "args": args,
                    "payload": payload,
                    "result": tool_result,
                })

                # tool 결과를 role=tool로 LLM에 피드백
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": f"tc_{i}",
                    "name": func_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })
                self.logger.info(f"[TOOLCALL {i}] appended tool message")

            # 이번 턴의 tool 결과 채우고, 턴 디버그 리스트에 추가
            turn_info["tool_results"] = tool_results_turn
            turns_debug.append(turn_info)

            # 모든 MCP 호출 성공/실패 여부에 상관없이 후속 답변 요청
            if all_ok:
                self.logger.info("[BRIDGE] All tool calls OK; requesting follow-up answer from LLM.")
                continue
            else:
                self.logger.info("[BRIDGE] Some tool calls failed; still asking LLM for follow-up.")
                continue

        self.logger.warning("[CLIENT] Max turns reached without final answer.")
        if debug:
            return {
                "final_answer": assistant_text,
                "turns": turns_debug,
                "warning": "max_turns_reached",
            }
        return assistant_text
