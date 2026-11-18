# 洛谷、LeetCode、AtCoder 只读工具集

## Context

目标是在 `D:\学习资源\算竞` 中复用现有 `codeforces_tools.py` 的 Tool 包装方式，为另外三个 OJ 提供用户数据、题目列表和确定性题目推荐。工具只读，不登录、不提交代码、不绕过验证码或访问限制；不修改 promptkit 核心、现有 Codeforces 工具及用户记忆/会话数据。本轮只交付可直接挂载的工具，不改动 `算竞智能体.py`。

已验证：

- 洛谷页面内 `lentille-context` JSON 可返回题库和公开练习记录；私密用户的通过数为 `null`，不能误判为零 AC。
- LeetCode `.com` GraphQL 可匿名查询公开用户统计、题库和近期 AC；`.cn` 不支持相同 schema，本期仅支持 `.com`。
- AtCoder Problems API 可查询题库、难度模型、提交与排名；它是非官方 API，请求间隔需大于 1 秒。

## 实现

### 1. 新增 `D:\学习资源\算竞\luogu_tools.py`

提供 `luogu_tools(uid=0, timeout=25.0) -> List[Tool]`：

- `lg_user`：公开资料、排名、通过/尝试题数、练习记录是否可见。
- `lg_problems`：按页读取题目，返回题号、题名、难度、原始标签 ID、提交/通过数、通过率和链接。
- `lg_recommend`：排除公开 `passed` 中的题，按难度区间过滤，最多扫描 3 页，按难度、通过率、题号确定性排序。

使用标准库 `HTMLParser` 提取 `<script id="lentille-context">` 中的 JSON，不依赖 BeautifulSoup。题目页缓存 1 小时，用户练习缓存 5 分钟。私密历史用于推荐时给出明确错误，不将其当成新用户。

### 2. 新增 `D:\学习资源\算竞\leetcode_tools.py`

提供 `leetcode_tools(username="", timeout=25.0) -> List[Tool]`：

- `lc_user`：用户名、ranking、reputation、各难度 AC 统计。
- `lc_problems`：GraphQL 分页和难度过滤，返回题号、题名、slug、难度、通过率、标签、是否付费及链接。
- `lc_recommend`：默认排除付费题，并避开 `recentAcSubmissionList` 返回的近期公开 AC；支持难度与起始 offset，最多扫描 4 页。

明确返回 `history_complete=False` 等元数据：匿名接口不能取得完整历史，推荐不能声称“全历史未 AC”。GraphQL errors、空用户和结构变化均转成清晰异常。题目页缓存 1 小时，用户/近期 AC 缓存 5 分钟。

### 3. 新增 `D:\学习资源\算竞\atcoder_tools.py`

提供 `atcoder_tools(username="", timeout=25.0, max_submission_pages=4) -> List[Tool]`：

- `at_user`：AC 数、rated point sum、最长 streak 及对应 rank。
- `at_problems`：合并 `problems.json` 与 `problem-models.json`，支持 offset、比赛和原始估计难度区间过滤。
- `at_recommend`：读取指定时间窗内提交（默认 365 天），排除窗口内 AC，支持比赛/难度过滤并返回推荐。

提交按 `epoch_second + 1` 前进，每页最多 500 条，默认最多 4 页；返回 `from_second/pages_fetched/submissions_checked/truncated/history_scope`，不声称覆盖全历史。所有请求共享至少 1.1 秒间隔，题库和难度模型缓存 6 小时，用户数据缓存 5 分钟。

### 4. 共同约束与测试

三个模块均：

- 复用 `promptkit.core.tool.Tool` 和现有手工 JSON Schema 风格，返回摘要 dict/list，不返回大段原始响应。
- 仅使用 Python 标准库与 promptkit；完整类型注解、工厂 docstring 和模块内使用示例。
- 默认用户名支持显式工厂参数或 `LUOGU_UID`、`LEETCODE_USERNAME`、`ATCODER_USERNAME` 环境变量；代码中不写本机绝对路径或凭据。
- 对页码、数量、难度区间和缺失用户名做边界校验；HTTP 403/429/5xx 不自动重试，不绕过限制。
- 不做后台任务、并发抓取、登录/Cookie 管理或持久化缓存。

新增 `D:\学习资源\算竞\tests\test_online_judge_tools.py`，集中覆盖三个平台的离线 mock 测试：Tool schema/默认参数、洛谷嵌入 JSON 与隐私语义、LeetCode GraphQL 错误与近期 AC 过滤、AtCoder 分页/截断/限流，以及推荐结果的过滤关系和历史范围元数据。

## 验证

1. 运行离线 pytest，确保测试不访问网络。
2. 对新增模块运行语法编译和可用的 mypy strict 检查。
3. 用公开样例做真实 smoke test：洛谷 UID `884897`、LeetCode 用户 `lee215`、AtCoder 用户 `chokudai`；分别调用用户、题目列表和少量推荐，只断言字段、过滤关系、链接和历史范围，不断言实时数值。
4. 检查 9 个工具名称无冲突、Schema 可 JSON 序列化，且现有 `codeforces_tools.py` 与 `算竞智能体.py` 未改变。

## 接口参考

- AtCoder Problems API：https://github.com/kenkoooo/AtCoderProblems/blob/master/doc/api.md
- 洛谷题库入口：https://www.luogu.com.cn/problem/list
- LeetCode GraphQL：https://leetcode.com/graphql
