# -*- coding: utf-8 -*-
"""把 Claude Code 的会话记录（~/.claude/projects 下的 .jsonl）导出成可读 Markdown，或跨会话检索。

上下文压缩（compact）只影响模型能看到的内容，磁盘上的记录始终是完整原文，用这个脚本取回。

用法:
    python Claude/dump_chat.py --list                   # 列出本项目的所有会话（按时间倒序）
    python Claude/dump_chat.py <会话id>                 # 导出到 Claude/chat/<会话id>.md
    python Claude/dump_chat.py <会话id> 输出.md         # 指定输出路径
    python Claude/dump_chat.py --grep 静止假设          # 在所有会话里搜关键词，打印命中的消息
选项:
    --project-dir <目录>   指定会话目录（默认本项目；找不到时取 ~/.claude/projects 下最新的一个）
    --with-tools           导出时保留工具调用与结果的摘要（默认只保留对话文本）
"""
import argparse
import datetime
import json
import os
import sys

try:                                     # Windows 控制台默认不是 UTF-8，中文会显示成乱码
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))     # 导出统一放在本脚本同级的 chat/ 下
ROOT = os.path.expanduser("~/.claude/projects")
DEFAULT_PROJECT = os.path.join(ROOT, "d-----D-EV-UAV-EV-UAV")


def project_dir(override):
    """会话目录：命令行指定 > 本项目默认 > ~/.claude/projects 下最近修改的目录。"""
    if override:
        return override
    if os.path.isdir(DEFAULT_PROJECT):
        return DEFAULT_PROJECT
    dirs = [os.path.join(ROOT, n) for n in os.listdir(ROOT)] if os.path.isdir(ROOT) else []
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        raise SystemExit("找不到会话目录：%s" % ROOT)
    return max(dirs, key=os.path.getmtime)


def sessions(directory):
    """[(修改时间, 路径)]，按时间倒序。"""
    rows = [(os.path.getmtime(os.path.join(directory, n)), os.path.join(directory, n))
            for n in os.listdir(directory) if n.endswith(".jsonl")]
    return sorted(rows, reverse=True)


def text_of(message, with_tools):
    """把一条消息的 content 拉平成纯文本；工具调用与结果只留摘要。"""
    content = message.get("content")
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind == "thinking" and with_tools:
            parts.append("_[思考]_")
        elif kind == "tool_use" and with_tools:
            parts.append("`[工具 %s]`" % block.get("name"))
        elif kind == "tool_result" and with_tools:
            body = block.get("content")
            body = body if isinstance(body, str) else "(结构化结果)"
            parts.append("`[工具结果]` " + body[:300].replace("\n", " "))
    return "\n".join(p for p in parts if p)


def messages(path, with_tools):
    """逐条产出 (角色, 时间戳, 正文)，跳过空消息与非对话记录。"""
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("type") not in ("user", "assistant"):
                continue
            body = text_of(record.get("message", {}), with_tools).strip()
            if body:
                yield record["type"], record.get("timestamp", "")[:19], body


def resolve(directory, name):
    """会话 id / 文件名 / 完整路径 -> 文件路径。"""
    for candidate in (name, os.path.join(directory, name),
                      os.path.join(directory, name + ".jsonl")):
        if os.path.isfile(candidate):
            return candidate
    raise SystemExit("找不到会话：%s（用 --list 查看）" % name)


def main():
    parser = argparse.ArgumentParser(description="导出 / 检索 Claude Code 会话记录")
    parser.add_argument("session", nargs="?", help="会话 id 或文件名")
    parser.add_argument("out", nargs="?", help="输出的 Markdown 路径")
    parser.add_argument("--list", action="store_true", help="列出所有会话")
    parser.add_argument("--grep", help="在所有会话里搜关键词")
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--with-tools", action="store_true", help="保留工具调用与结果摘要")
    args = parser.parse_args()
    directory = project_dir(args.project_dir)

    if args.list or (not args.session and not args.grep):
        print("会话目录:", directory)
        for mtime, path in sessions(directory):
            stamp = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            print("%s  %6.1f MB  %s" % (stamp, os.path.getsize(path) / 1e6,
                                        os.path.basename(path)[:-6]))
        return

    if args.grep:
        needle, hits = args.grep, 0
        for _, path in sessions(directory):
            for role, stamp, body in messages(path, args.with_tools):
                if needle in body:
                    hits += 1
                    where = body.index(needle)
                    snippet = body[max(0, where - 120):where + 200].replace("\n", " ")
                    print("[%s] %-9s %s\n    …%s…\n" % (stamp, role, os.path.basename(path)[:8], snippet))
        print("命中 %d 条" % hits)
        return

    src = resolve(directory, args.session)
    dst = args.out or os.path.join(HERE, "chat", os.path.basename(src)[:-6] + ".md")
    if os.path.dirname(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
    rows = ["## %s  `%s`\n\n%s\n" % ("我" if role == "user" else "Claude", stamp, body)
            for role, stamp, body in messages(src, args.with_tools)]
    with open(dst, "w", encoding="utf-8") as stream:
        stream.write("\n".join(rows))
    print("导出 %d 条消息 -> %s（%.1f KB）" % (len(rows), dst, os.path.getsize(dst) / 1e3))


if __name__ == "__main__":
    main()
