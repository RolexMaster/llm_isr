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

                # 기본 payload는 그냥 args로 설정 (mcp_name이 없을 때도 사용)
                payload = args

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
                            if (getattr(t, "name", None) or t.get("name"))
                            == mcp_name
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
                        if not tool_result or not tool_result.get("ok", False):
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
                        "content": json.dumps(
                            tool_result, ensure_ascii=False
                        ),
                    }
                )
                self.logger.info(f"[TOOLCALL {i}] appended tool message")

            # 이번 턴의 tool 결과 채우고, 턴 디버그 리스트에 추가
            turn_info["tool_results"] = tool_results_turn
            turns_debug.append(turn_info)

            # 모든 MCP 호출 성공/실패 여부에 상관없이 후속 답변 요청
            if all_ok:
                self.logger.info(
                    "[BRIDGE] All tool calls OK; requesting follow-up answer from LLM."
                )
                continue
            else:
                self.logger.info(
                    "[BRIDGE] Some tool calls failed; still asking LLM for follow-up."
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
