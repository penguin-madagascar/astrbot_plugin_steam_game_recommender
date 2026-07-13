# astrbot_plugin_steam_game_recommender

Steam 游戏推荐助手。插件根据自然语言偏好召回 Steam 游戏，用 0–100 的连续分数排序，并为每款游戏生成 2–3 句基于可信证据的精简理由。

默认无需游戏数据 API Key。安装同级插件 `astrbot_plugin_steam_price_heybox` 后，可额外查询用户指定区域的当前价格、历史最低和最近一次促销。

## 功能

- 仅推荐和验证 Steam 商店游戏；普通游戏请求默认按 Steam 查询。
- 根据玩法标签、参考游戏、已玩游戏画像、贝叶斯口碑、知名度和数据完整度计算百分比推荐分。
- Steam 语言列表明确区分支持、不支持和数据未知；简体中文与繁体中文分别验证。
- 支持正向参考、负向参考、排除标签、人数、预算、语言、难度、氛围及游戏库过滤。
- 支持查询级区域参数；价格只查询所选区域，不请求全球区价。
- 每款游戏独立生成简短 LLM 理由，最多 5 路并发；格式、证据 ID 或重要风险校验失败时仅该款降级为规则理由。
- 保存最近 30 分钟的推荐结果与反馈，可按补充条件换一批。
- 可随机推荐 Steam 游戏库中尚未游玩且评价达到门槛的游戏。

## 命令

- `/gamerec <自然语言需求>`：推荐 Steam 游戏；alias：`/游戏推荐`。
- `/gamerec_retry [补充要求]`：基于最近一次结果换一批；alias：`/重新推荐`、`/换一批`。
- `/accountbind <SteamID64|好友码>`：绑定当前聊天用户的 Steam 账号；alias：`/账号绑定`。
- `/randomrec`：随机选择一款未玩且评价过线的库内游戏；alias：`/随机推荐`。

`/gamerec` 还支持以下前置游戏库过滤参数：

- `排除已有` / `exclude-owned`
- `仅查看已有` / `only-owned`

两个参数互斥，并依赖 `steam_api_key`、账号绑定和公开可读的 Steam 游戏库。

## 区域与预算

可使用两字母区域代码或常用中文写法：

```text
/gamerec -US 双人合作解谜，预算 $30
/gamerec 日区 轻松剧情游戏，预算 3000 日元
/游戏推荐 国区 类似星露谷物语的种田经营，预算 100 以内
```

支持 `-CN`、`-US`、`-JP` 等代码，以及国区、美区、日区、港区、台区、韩区等写法。未指定时使用 `default_region`。

预算没有显式币种时，按所选区域本币解释；显式币种与查询价格币种不一致时，不调整推荐分。

预算对分数的调整为：

- 当前价在预算内：`+5`
- 当前价和历史最低都高于预算：`-5`
- 只有历史最低进入过预算：不调整
- 当前价格或必要历史数据未知：`-2`
- 币种不一致：不调整

## 返回格式

开场白固定为：

```text
找到 3 款 Steam 游戏，按推荐分从高到低排列。
```

只有确实存在解析或过滤提示时，开场白后才会增加一行提示。每款游戏格式如下：

```text
1. 《示例游戏》｜推荐分：86%
推荐理由：玩法契合合作解谜偏好。Steam 口碑稳定，但简体中文支持尚未确认。
价格（US）：当前价 $19.99；历史最低 $9.99；最近促销 $11.99（结束于 18 天前）
购买链接：https://store.steampowered.com/app/123456/
```

价格行仅包含查询区域的当前价、历史最低、最近促销价和距促销的时间。购买链接仅提供 Steam 商店链接。

`/randomrec` 使用普通单条消息，只返回游戏名和 2–3 句玩法、口碑、知名度理由，不返回分数、价格或链接。

## 评分规则

正向分由以下部分组成：

- 标签覆盖：50%
- 正向参考：15%
- Steam 游戏库画像：10%
- 贝叶斯口碑：10%
- 知名度：10%
- 数据完整度：5%

没有提供正向参考或无法读取游戏库画像时，对应项会移除，其余适用权重按比例归一化。知名度使用：

```text
min(log10(Steam 评测数 + 1) / 5, 1)
```

与负向参考的相似度最多扣 20 分，未知硬条件最多扣 15 分；确认违反硬条件的候选直接过滤。最终分限制在 0–100 并取整数，再依次按推荐分、标签覆盖、评测数、发售年份和标题排序。

## LLM 行为

LLM 用于偏好解析和精简推荐理由。推荐理由只能引用输入的编号化可信证据，输出必须包含 `appid`、`reason` 和 `evidence_ids`；重要风险不能省略。

单款理由调用失败、格式不合法、超过 180 字、不是 2–3 句、引用未知证据或遗漏重要风险时，只对该款使用规则理由。

`enable_llm_fallback` 仅控制未验证候选兜底：开启后，Steam 索引无结果或用户明确要求非 Steam 平台时，可返回清楚标注的 LLM 未验证建议；关闭时直接返回范围或空结果提示。

## 安装

1. 将目录命名为 `astrbot_plugin_steam_game_recommender` 并放入 AstrBot 插件目录，或通过插件管理安装。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 直接使用 `/gamerec`。如需游戏库过滤或 `/randomrec`，再配置 `steam_api_key` 并绑定账号。

## 配置

- `llm_provider_id`：偏好解析和推荐理由使用的模型；留空时尝试当前会话模型。
- `enable_llm_fallback`：是否允许生成已标注的未验证候选，默认关闭。
- `default_region`：默认 Steam 查询区域，默认 `CN`。
- `steam_api_key`：用于读取已绑定账号的 Steam 游戏库。
- `max_results`：默认推荐数量上限，范围 1–10。
- `steam_index_ttl_hours`：Steam 推荐索引缓存有效期，默认 168 小时。
- `steam_min_review_count`：口碑贝叶斯先验强度，同时是 `/randomrec` 最低评测数。
- `steam_min_positive_ratio`：`/randomrec` 最低好评率，默认 0.65。
- `cache_ttl_hours`：Steam 查询缓存有效期。
- `timeout_seconds`：Steam 与价格请求超时时间。
- `steam_price_heybox_notice`：只读提示；安装价格插件后自动启用指定区域价格。

## 限制

- 价格依赖可导入的 `astrbot_plugin_steam_price_heybox`；未安装或查询失败时，价格三项显示暂无数据。
- 游戏库功能依赖有效的 Steam Web API Key 和公开可读的游戏库。
- `/randomrec` 将 `playtime_forever` 为 0 的条目视为未游玩。
- Steam 商店语言字段缺失时只标记未知，不会推断为支持或不支持。
- 多人、合作、难度和氛围主要依据 Steam 分类、标签与描述，数据可能不完整。

## 开发验证

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -r ../astrbot_plugin_steam_price_heybox/requirements.txt pyyaml
PYTHONPATH=.. .venv/bin/python -m compileall -q .
PYTHONPATH=.. .venv/bin/python -m unittest discover -s tests
uvx ruff check .
uvx ruff format --check .
git diff --check
```
