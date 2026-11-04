# Windows VS Code → SSH Linux 开发节点 → 离线 GLM 编程

适用：本地 Windows10/11 x86_64；开发节点按此前机器准备 Linux x86_64；GLM5.1 部署在另一个节点。本包不部署/训练 GLM，不假定它的实际 API 模型 ID。Windows 不需要 GPU。

## 下载什么

[Windows x64 VS Code 1.145.0 官方安装程序](https://vscode.download.prss.microsoft.com/dbazure/download/stable/a717861c07586a53d83b80d742cd3f59b20a6273/VSCodeUserSetup-x64-1.145.0.exe)

本配置包提供固定版本官方下载清单、离线 Server 安装器、SSH/Continue 配置、API 检查脚本。**配置包本身不含约565 MB的安装二进制**；这些文件由脚本直接从 Microsoft Marketplace / Microsoft CDN / Open VSX 下载并核对 SHA256。

在能上网的 Windows 电脑解压本配置包，然后在 PowerShell 运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\vscode-glm-kit\scripts\download-windows.ps1
```

仅为本次脚本进程设置执行策略。默认保存到桌面 `SAR-VSCode-Offline`，复制整个目录进入内网。也可用参数 `-OutputDirectory D:\SAR-VSCode-Offline`。脚本可跳过已经下载且哈希正确的文件。下载清单中所有包已实际下载核验，详见下载清单版本和 SHA256。

文件用途：

| 文件 | 安装位置 |
| --- | --- |
| VSCodeUserSetup-x64-1.145.0.exe | Windows 本机 |
| remote-ssh-0.122.0.vsix、remote-ssh-edit-0.87.0.vsix、remote-explorer-0.5.0.vsix | Windows 的 VS Code |
| vscode-server-linux-x64-提交号.tar.gz | Linux x86_64 开发节点 |
| Continue.continue-1.3.40-linux-x64.vsix | Linux x86_64 开发节点的远程 VS Code 扩展目录，不能安装为 Windows 本地扩展 |

这里固定使用 Continue 1.3.40 的配置格式。VS Code 桌面与 Server 必须使用清单中相同提交号；若更新桌面版本，需同时准备对应 Server。开发节点若实际是 ARM64，需要换 ARM64 Server 和 Continue VSIX，此处的 Windows 安装包仍不变。

## 1. Windows 安装与 SSH

双击 EXE，建议勾选 Add to PATH，完成后重新打开 PowerShell：

```powershell
code --version
ssh -V
```

若没有 ssh，先让管理员启用匹配 Windows 版本的 OpenSSH Client 可选功能。VS Code 本身不提供 SSH 登录权限。

在离线目录执行，先装附属扩展，再装 Remote SSH：

```powershell
code --install-extension .\remote-ssh-edit-0.87.0.vsix
code --install-extension .\remote-explorer-0.5.0.vsix
code --install-extension .\remote-ssh-0.122.0.vsix
```

将 `configs/ssh-config.example` 合并到 Windows 的 `%USERPROFILE%\.ssh\config`，替换开发节点 IP、用户名和端口。密码登录可删去 IdentityFile 行；如使用密钥，先在 Windows 创建密钥，再由自己或管理员把公钥放入开发节点的 authorized_keys：

```powershell
ssh-keygen -t ed25519
ssh sar-dev
```

首次连接按实际服务器指纹核实主机身份。若经过登录节点，用配置中的 ProxyJump 示例。开发节点和模型节点不必是同一台；不要把 GLM API 端口误写为 SSH Port。计算集群有调度规则时，应连接获准的节点或交互作业分配的节点。

## 2. 离线安装 Linux VS Code Server

先在 Linux 开发节点查看：

```bash
uname -m
getconf GNU_LIBC_VERSION
```

此包要求 x86_64。新版 VS Code 官方 Linux 远端通常要求 kernel>=4.18、glibc>=2.28、libstdc++ 的 GLIBCXX_3.4.25 或更新。麒麟 V10 的具体更新级别需实测；不要为此直接覆盖系统 libc。安装器会先尝试运行实际 Server 的 node/code-server，不能执行时即报错。

将 Server 压缩包和配置包目录传到开发节点。例如从 Windows PowerShell：

```powershell
ssh sar-dev "mkdir -p ~/vscode-offline"
scp .\vscode-server-linux-x64-a717861c07586a53d83b80d742cd3f59b20a6273.tar.gz sar-dev:~/vscode-offline/
scp .\Continue.continue-1.3.40-linux-x64.vsix sar-dev:~/vscode-offline/
scp -r .\vscode-glm-kit sar-dev:~/vscode-offline/
```

在开发节点以自己的普通 SSH 用户执行（不用 sudo）：

```bash
bash ~/vscode-offline/vscode-glm-kit/scripts/install_server.sh \
  ~/vscode-offline/vscode-server-linux-x64-a717861c07586a53d83b80d742cd3f59b20a6273.tar.gz
```

在 Windows VS Code 按 Ctrl+Shift+P → `Preferences: Open User Settings (JSON)`，将 `configs/windows-settings.fragment.json` 的各项合并到现有设置，不覆盖你的其他设置。这里设定 useExecServer=false，使用上述 `~/.vscode-server/bin/提交号` 离线路径；禁用自动升级以保持版本匹配；将 Continue 指定为 workspace 扩展在远端运行。

然后 Ctrl+Shift+P → `Remote-SSH: Connect to Host...` → `sar-dev`。左下角显示 `SSH: sar-dev` 后，在远程窗口打开服务器项目目录，例如 `/srv/code/SAR`。后续终端、读写项目文件均发生在开发节点。

## 3. 将 Continue 安装到开发节点

必须在已连接 `SSH: sar-dev` 的 VS Code 集成终端中执行（该终端会提供远程 code 命令）：

```bash
code --install-extension ~/vscode-offline/Continue.continue-1.3.40-linux-x64.vsix
```

检查扩展面板显示 Continue 安装在 `SSH: sar-dev`，然后 Reload Window。不要在 Windows 本地安装 Linux VSIX。配置中的 remote.extensionKind 用来确保请求从开发节点发出，而不是从 Windows 发出。

## 4. 先验证 GLM API

向 GLM 部署管理员取得：开发节点可访问的 API Base URL、准确模型 ID、是否需要 API Key、是否采用 OpenAI Chat Completions 协议。**“GLM5.1”是你提供的部署名称，不能据此猜测 model 字段。** 示例端口8000只供替换。

在开发节点运行：

```bash
python3 ~/vscode-offline/vscode-glm-kit/scripts/check_glm_api.py \
  --base-url http://实际GLM节点IP:实际端口/v1 --no-auth
```

无认证服务才用 `--no-auth`；有认证时删掉它，脚本隐藏输入 Key，或读取环境变量 GLM_API_KEY。不要把真实 Key 放进 GitHub 或本项目共享文件。

取 `/models` 返回的准确 ID，进一步检查聊天和结构化工具调用：

```bash
python3 ~/vscode-offline/vscode-glm-kit/scripts/check_glm_api.py \
  --base-url http://实际GLM节点IP:实际端口/v1 \
  --model 实际模型ID --check-tools --no-auth
```

若 /models 不提供，可直接用管理员给的 --model。Base URL 通常截至 `/v1`，不要填成 `/v1/chat/completions`，以免重复拼接路径。HTTP401/403 是认证/访问控制，404 是路径或模型错误；无法建立连接需检查从**开发节点到 GLM 节点**的路由、监听地址和防火墙。内网私有 HTTPS CA 应正确安装信任，不通过关闭 TLS 校验解决。

Chat PASS 表示真实接口聊天可用。工具调用探测只要求返回一个不执行的 report_probe 工具调用：PASS 不代表完整 Agent 已验证；不支持 forced tool_choice 的网关也可能探测失败，应先用聊天/编辑方式工作并联系部署管理员确认。服务需兼容聊天及流式响应；Agent 还要求结构化 tool_calls、工具结果回传等完整协议。

## 5. 配置 Continue

在开发节点准备个人配置目录：

```bash
mkdir -p ~/.continue
chmod 700 ~/.continue
```

先备份已有 `~/.continue/config.yaml`。将 `configs/config.chat.yaml` 复制为该文件，替换：

```yaml
model: 实际模型ID
apiBase: http://实际GLM节点IP:实际端口/v1
apiKey: 实际Key
```

无认证服务可填 `apiKey: EMPTY`。Key 只写个人配置文件，设置 `chmod 600 ~/.continue/config.yaml`，不要放在项目目录。Reload Window，在 Continue 中选择 `GLM private endpoint`。不需要登录云端模型服务，也不需要将 GLM 权重复制到 Windows。

先用 Chat 提问和 Edit/Apply 修改选中文件。确实确认 API 支持工具调用后，可改用 `config.agent.yaml` 模板（同样替换地址、模型 ID、Key）；它添加 `capabilities: [tool_use]`，这只是向插件声明能力，不能让不支持的服务自动具备工具调用。

## 6. 对话修改代码的操作方式

1. 打开远程项目，先执行 `git status`，有需要时创建工作分支。
2. Continue Chat 选择相关文件作为上下文，例如：`阅读这个文件，解释数据读取流程和失败处理，不修改代码。`
3. Edit/Apply 请求：`给读取 CSV 的函数添加空字段校验，保持现有接口，展示差异并运行相应单元测试。` 查看 Diff 后应用。
4. Agent（协议已验证后）示例：`为SAR批量评估增加每类召回率CSV导出。先检查现有实现，修改必要文件，运行已有评估测试，最后列出改动和真实测试结果。不要启动训练或提交主分支。`

Agent 的终端命令按需审批，确认真实调用了工具、产生文件 Diff 和测试日志。普通聊天回复一段代码不会自动等于已写入文件。纯内网工作不需要添加互联网搜索/MCP 服务或默认云端 embedding；本模板不配置 autocomplete/embed/rerank，避免额外下载模型。上下文先选相关文件，不一次塞入完整数据集或模型权重。

开发环境是之前 Docker 时，可先把项目目录挂载到容器，再从 VS Code 远程终端 `docker exec -it 实际容器名 bash` 运行测试；编辑的是服务器项目目录，容器挂载后读取同一份文件。无需为了 API 对话改动训练镜像。只有以后需要容器内 Python 调试/语言服务时，再按需安装 Dev Containers 等扩展。

## 验证范围与资料

下载清单所有二进制已实际下载、SHA256核验，VS Code EXE 与 Microsoft API 的 SHA256 一致；Continue 与 Open VSX 发布摘要一致；VSIX 已读取清单核查目标平台和版本。Server 的原生运行及 API 检查脚本用模拟 HTTP 服务验证结果见随附 VALIDATION.md/日志。**未连接你的实际 Windows、开发节点或 GLM 服务，真实 SSH/API/Agent 端到端测试需你按上述步骤执行。**

官方参考：
- https://code.visualstudio.com/docs/remote/ssh
- https://code.visualstudio.com/docs/remote/linux
- https://docs.continue.dev/reference
- https://docs.continue.dev/customize/model-providers/top-level/openai
