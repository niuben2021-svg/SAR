# 完整离线包：从这里开始

这个完整包已经包含 Windows VS Code 安装程序、三个本地 Remote SSH 相关 VSIX、匹配的 Linux x86_64 VS Code Server、Linux x86_64 Continue VSIX、配置模板和 API 检查脚本。无需再运行 download-windows.ps1 或访问外网下载这些文件。

本地电脑：Windows10/11 x86_64。开发节点：Linux x86_64。GLM 是你在另一节点部署的私有服务。没有模型权重或真实 Key 被打包。

## 1. 将整个包解压到内网 Windows

例如解压后目录为 `D:\SAR-VSCode-Offline`，其中应直接看到 EXE、VSIX、Server tar.gz、vscode-glm-kit 文件夹。路径可自行选择，不要只拷走 EXE。

在 PowerShell 校验七个二进制文件：

```powershell
cd D:\SAR-VSCode-Offline
$m = Get-Content .\vscode-glm-kit\downloads.json -Raw | ConvertFrom-Json
foreach ($f in $m.files) {
  if ((Get-FileHash $f.name -Algorithm SHA256).Hash.ToLower() -ne $f.sha256) { throw "文件校验失败: $($f.name)" }
}
```

## 2. 安装 Windows 软件和三个本地扩展

双击 `VSCodeUserSetup-x64-1.145.0.exe`，安装时建议勾选 Add to PATH。重新打开 PowerShell，运行：

```powershell
code --version
code --install-extension .\remote-ssh-edit-0.87.0.vsix
code --install-extension .\remote-explorer-0.5.0.vsix
code --install-extension .\remote-ssh-0.122.0.vsix
ssh -V
```

如果安装时没有添加 PATH，可使用 `%LOCALAPPDATA%\Programs\Microsoft VS Code\bin\code.cmd` 的完整路径执行这些命令。

Windows10/11 通常已有 OpenSSH 客户端。如果 `ssh -V` 找不到命令，本包另含 Microsoft Win32-OpenSSH 官方 ZIP 作为备用：

```powershell
Expand-Archive .\OpenSSH-Win64.zip -DestinationPath .\ssh-client
.\ssh-client\OpenSSH-Win64\ssh.exe -V
```

上游将备用版本 10.4.0.0p1 标记为 Preview；优先使用系统现有 SSH。这里只需要 ssh.exe/scp.exe/ssh-keygen.exe 客户端，**无需运行 install-sshd.ps1 或安装 Windows SSH 服务**。若使用备用客户端，在 Windows VS Code 用户设置中另加（改成实际路径）：

```json
"remote.SSH.path": "D:\\SAR-VSCode-Offline\\ssh-client\\OpenSSH-Win64\\ssh.exe"
```

命令行 ssh/scp 也要使用其完整路径，或仅在当前 PowerShell 中把这个目录加入 `$env:Path`。

## 3. SSH 配置和离线 Server

打开 `vscode-glm-kit/README.md`，从“1. Windows 安装与 SSH”开始按步骤完成：

- 用 `configs/ssh-config.example` 配置开发节点 IP、端口、用户名、密钥/密码；先 `ssh sar-dev` 验证内网 SSH。
- 将匹配的 `vscode-server-linux-x64-a717861c07586a53d83b80d742cd3f59b20a6273.tar.gz`、Continue VSIX 和 `vscode-glm-kit` 文件夹传到开发节点。
- 在开发节点运行 `scripts/install_server.sh`。
- 将 `configs/windows-settings.fragment.json` 合并到 Windows 用户设置，确保离线 Server 路径匹配、Continue 在远端运行、版本不自动升级。
- Remote-SSH 连接开发节点，在远程终端安装 **Linux** Continue VSIX，再打开项目目录。

Linux Server 和 Continue 文件是给开发节点的，不能作为 Windows 本地扩展安装。GLM 节点本身不需要安装 VS Code。

## 4. 配置离线 GLM

在开发节点运行 `scripts/check_glm_api.py`，用真实 API Base URL 查询实际模型 ID，并验证聊天/工具调用。然后编辑开发节点自己的 `~/.continue/config.yaml`，填入地址、模型 ID、必要的 Key。模板在 configs/ 下，不包含真实访问参数。

网络关系：Windows → 内网 SSH → 开发节点 → 内网 API → GLM 节点。无需互联网；但 SSH 和 API 路由、账号权限需要有效。Agent 自动读写文件和执行测试还要求 GLM 服务兼容工具调用，普通聊天接口成功不能替代该验证。

## 验证范围

请查看 validation/ 随附日志：官方文件 SHA256、Windows Microsoft 数字签名、备用 SSH 客户端版本运行、Server 版本/安装器和 API 探测脚本的模拟测试。没有实际连接你的机器或真实 GLM；Windows GUI 安装、用户节点兼容性及真实 Agent 工作仍需现场验证。
