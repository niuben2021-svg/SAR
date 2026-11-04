# 验证范围

- 所有六个官方二进制下载已完成，并生成 SHA256；EXE 摘要与 Microsoft 更新 API 一致，Continue 摘要与 Open VSX 发布值一致。
- VSIX 清单确认 Remote SSH 0.122.0、附属扩展 0.87.0/0.5.0，Continue 1.3.40 目标为 linux-x64、extensionKind 为 ui/workspace。
- API 检查脚本的模拟 HTTP 回归：3 项通过，包含聊天成功、工具调用成功、纯文本不能冒充工具调用；测试检查 API Key 不在输出中出现。模拟测试不是 GLM 的实测。
- 本地 Linux 原生 Server 版本命令输出 VS Code 1.145.0、固定提交 a717861c07586a53d83b80d742cd3f59b20a6273。首次直接探测受执行环境缺失默认 home 路径影响，改用临时 VSCODE_AGENT_FOLDER 后复查通过。
- GitHub 工作流在 Windows 运行真实 PowerShell 下载脚本并检查 EXE Authenticode，在 Linux 检查原生 Server 安装器；构建成功才发布配置包，实际结果见 validation/ 随附日志。
- 用户 Windows GUI 安装、真实 SSH 会话、麒麟 V10 二进制兼容性、真实 GLM API/流式输出/Agent 多轮工具调用均 NOT RUN。需要用户按 README 对实际节点检查。
- 本包不包含真实 IP、用户名或 API Key，不修改已有训练镜像或主分支。
