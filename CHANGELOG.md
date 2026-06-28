# Changelog

## Unreleased

- 新增通用参考游戏归一：支持“类似/像/接近 X”表达，通过别名索引、RAWG 标题 fallback、置信度评分和可维护画像解析参考游戏。
- 改进推荐召回：在宽泛 RAWG 标签查询前优先执行参考画像种子搜索，避免相似合作游戏被无关高分游戏挤掉。
- 将 `/gamerec` 输出拆分为一条开场白消息 + 每个推荐游戏一条消息。
- 将拆分后的推荐消息包装为一个新的合并转发聊天记录发送，避免在群聊内刷出多条普通消息。
- 改进推荐解释：输出具体适合点、平台 caveat 和明确不确定项，不再使用空泛的“暂未发现明显不适合点”。
- 重构推荐引擎为约束优先流程：使用 RAWG 召回、Steam 商店分类/语言事实和价格增强前候选池，按强烈推荐/推荐/备选分层输出，防止全库高分榜和普通多人标签压过双人合作硬约束。

## 0.3.1 - 2026-06-28

- Improved keyword preference fallback for platform, two-player, budget, language, horror exclusion, difficulty, and reference-game hints.
- Strengthened filtering and de-duplication for DLC, single-player-only candidates, repeated editions, and disliked genres.
- Added Steam price enrichment through the optional sibling `astrbot_plugin_steam_price_heybox` plugin.
- Fixed recommendation reason formatting so values such as `RAWG 评分 4.8/5` are not split on `/`.
