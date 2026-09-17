"""
Dataset-Agnostic Prompt Templates for GIFT Portfolio Optimization

Designed for:
- Alibaba Bailian / Qwen-Plus
- Multiple experimental stock datasets
- Changing ticker symbols across W1/W2/W3/... datasets
- PPO portfolio optimization
- Stable LLM Python code generation

Main goals:
1. Never hard-code stock ticker symbols.
2. Make generated state code reusable across datasets.
3. Reduce LLM syntax failures.
4. Prevent undefined local variables.
5. Prevent hallucinated feature_library functions.
6. Force numerically safe NumPy-only feature generation.
7. Preserve all interfaces required by gift_controller.py.
"""

import json
import re


# ============================================================
# Utility
# ============================================================

def _fmt(val, fmt_str):
    """Safely format numeric values."""
    if isinstance(val, (int, float)):
        return format(val, fmt_str)
    return str(val)


# ============================================================
# JSON extraction
# ============================================================

def _extract_json(text: str) -> dict:
    """
    Extract one JSON object from an LLM response.

    Supports:
    1. Raw JSON.
    2. Markdown JSON blocks.
    3. JSON embedded inside surrounding text.
    """

    if not isinstance(text, str):
        raise TypeError("LLM response must be a string.")

    text = text.strip()

    # --------------------------------------------------------
    # 1. Direct raw JSON
    # --------------------------------------------------------

    try:
        obj = json.loads(text)

        if isinstance(obj, dict):
            return obj

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # 2. Markdown JSON blocks
    # --------------------------------------------------------

    pattern = r"```(?:json)?\s*(.*?)```"

    matches = re.findall(
        pattern,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    for match in matches:

        try:
            obj = json.loads(match.strip())

            if isinstance(obj, dict):
                return obj

        except json.JSONDecodeError:
            continue

    # --------------------------------------------------------
    # 3. Balanced JSON object search
    # --------------------------------------------------------

    depth = 0
    start = None
    in_string = False
    escaped = False

    for i, ch in enumerate(text):

        if escaped:
            escaped = False
            continue

        if ch == "\\" and in_string:
            escaped = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":

            if depth == 0:
                start = i

            depth += 1

        elif ch == "}":

            if depth > 0:
                depth -= 1

            if depth == 0 and start is not None:

                candidate = text[start:i + 1]

                try:
                    obj = json.loads(candidate)

                    if isinstance(obj, dict):
                        return obj

                except json.JSONDecodeError:
                    pass

                start = None

    raise ValueError(
        "Could not extract valid JSON from LLM response: "
        + text[:300]
    )


# ============================================================
# Python extraction
# ============================================================

def _extract_python_code(text: str) -> str:
    """
    Extract executable Python source from an LLM response.

    Generated code in this version is NumPy-only, therefore the
    preferred program starts with:

        import numpy as np

    This also avoids the old problem where feature_library import
    statements could be lost during extraction.
    """

    if not isinstance(text, str):
        raise TypeError("LLM response must be a string.")

    text = text.strip()

    # --------------------------------------------------------
    # 1. Handle markdown fences if Qwen ignores instructions
    # --------------------------------------------------------

    fenced_blocks = re.findall(
        r"```(?:python)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if fenced_blocks:

        valid_blocks = [
            block.strip()
            for block in fenced_blocks
            if "def revise_state" in block
            and "def intrinsic_reward" in block
        ]

        if valid_blocks:
            text = max(valid_blocks, key=len)

        else:
            text = max(
                fenced_blocks,
                key=len
            ).strip()

    # --------------------------------------------------------
    # 2. Find beginning of Python program
    # --------------------------------------------------------

    candidate_starts = []

    for token in (
        "import numpy as np",
        "import numpy",
        "def revise_state",
    ):

        index = text.find(token)

        if index >= 0:
            candidate_starts.append(index)

    if candidate_starts:

        text = text[min(candidate_starts):]

    # --------------------------------------------------------
    # 3. Remove accidental fences
    # --------------------------------------------------------

    text = text.replace("```python", "")
    text = text.replace("```", "")

    return text.strip()


# ============================================================
# Dataset-independent description
# ============================================================

DATASET_AGNOSTIC_DESC = """
============================================================
DATASET-INDEPENDENCE CONTRACT
============================================================

The exact financial assets vary between experiments.

Different experimental datasets may contain completely
different stock tickers.

Therefore:

- NEVER assume any fixed ticker symbol.
- NEVER hard-code any company name.
- NEVER hard-code TSLA.
- NEVER hard-code NFLX.
- NEVER hard-code AMZN.
- NEVER hard-code MSFT.
- NEVER hard-code JNJ.
- NEVER hard-code AAPL.
- NEVER hard-code NVDA.
- NEVER hard-code any other ticker.

The generated code must work without modification when the
experimental dataset is replaced by another group of assets.

Do NOT use company identity.

Do NOT use sector identity.

Do NOT create different formulas for different tickers.

Do NOT branch on asset names.

The function revise_state receives the historical numerical
state of ONE asset at a time.

Therefore every feature must be calculated exclusively from
that supplied numerical state.

CASH is handled separately by the portfolio environment.

The generated state-revision code should therefore transform
only the supplied risky-asset state.

Market statistics shown later may come from the current
experimental dataset.

They are allowed to guide the TYPE of feature that should be
useful, but asset names appearing in market statistics must
NEVER be converted into hard-coded Python logic.
"""


# ============================================================
# State layout
# ============================================================

STATE_LAYOUT_DESC = """
============================================================
INPUT STATE CONTRACT
============================================================

The input to revise_state is:

    s

where:

    type(s) is numpy.ndarray

and:

    s.shape == (120,)

The state represents exactly 20 historical trading days.

Each trading day contains six numerical channels in this order:

    close
    open
    high
    low
    volume
    adjusted_close


Therefore the following slices are valid:

    closes = s[0::6]

    opens = s[1::6]

    highs = s[2::6]

    lows = s[3::6]

    volumes = s[4::6]

    adjusted_closes = s[5::6]


Each slice contains exactly 20 observations.

Original dimensions occupy indices:

    0 through 119


The four generated dimensions must occupy:

    updated_s[120]
    updated_s[121]
    updated_s[122]
    updated_s[123]


Therefore:

    len(revise_state(s)) == 124

must ALWAYS be true.
"""


# ============================================================
# Extremely strict code contract
# ============================================================

STRICT_CODE_CONTRACT = """
============================================================
ABSOLUTE GENERATED-CODE CONTRACT
============================================================

This contract overrides all other suggestions.

The final response must be a SHORT executable Python program.


============================================================
1. ALLOWED PROGRAM STRUCTURE
============================================================

The program may contain ONLY:

    import numpy as np


    def revise_state(s):
        ...


    def intrinsic_reward(updated_s):
        ...


There must be exactly TWO functions.

Do NOT create helper functions.

Do NOT create classes.

Do NOT create decorators.

Do NOT create lambda functions.

Do NOT create generators.

Do NOT create nested functions.

Do NOT use try/except.

Do NOT open files.

Do NOT access networks.

Do NOT use external packages.


============================================================
2. IMPORT RULE
============================================================

The ONLY permitted import is:

    import numpy as np


DO NOT import:

    feature_library

DO NOT import functions from:

    feature_library


In particular, DO NOT use or import:

    compute_correlation_diversification_score

    compute_cross_sectional_rank

    compute_beta

or ANY other helper function.


If a financial feature is needed, implement the mathematical
formula directly using NumPy.


============================================================
3. CODE LENGTH RULE
============================================================

Keep generated code short.

Prefer:

    20 to 30 non-empty lines.

Never intentionally generate more than:

    40 non-empty lines.


Long generated programs are more likely to contain syntax
errors.

Avoid deeply nested expressions.

Avoid large multiline function calls.

Avoid unnecessary temporary variables.


============================================================
4. EXACT FEATURE COUNT
============================================================

revise_state MUST append EXACTLY FOUR scalar features.

Use exactly these generated-feature variable names:

    f1
    f2
    f3
    f4


Then construct:

    extra = np.array([f1, f2, f3, f4], dtype=np.float64)


Then sanitize:

    extra = np.nan_to_num(
        extra,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


Then return:

    return np.concatenate((x, extra))


The revised state must ALWAYS contain exactly:

    124 dimensions.


============================================================
5. VARIABLE-SCOPE RULE
============================================================

THIS RULE IS EXTREMELY IMPORTANT.

Variables declared inside revise_state are LOCAL.

They do NOT exist inside intrinsic_reward.


For example:

If revise_state contains:

    momentum = ...
    mr = ...
    volatility = ...


then intrinsic_reward MUST NOT use:

    momentum

    mr

    volatility


unless those variables are explicitly defined again inside
intrinsic_reward.


The safest required structure is:

    def intrinsic_reward(updated_s):

        f1 = float(updated_s[120])

        f2 = float(updated_s[121])

        f3 = float(updated_s[122])

        f4 = float(updated_s[123])

        phi = ...

        phi = float(
            np.nan_to_num(
                phi,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )
        )

        return phi


Inside intrinsic_reward prefer using ONLY:

    f1
    f2
    f3
    f4
    phi


NEVER reference a variable that was created only inside
revise_state.


============================================================
6. NUMERICAL SAFETY
============================================================

Every generated value must be finite.


Safe returns:

    returns = np.diff(closes) / (
        np.abs(closes[:-1]) + 1e-8
    )


Safe scalar denominator:

    numerator / (abs(denominator) + 1e-8)


Safe array denominator:

    numerator / (np.abs(denominator) + 1e-8)


Safe standard deviation:

    float(np.std(values)) + 1e-8


Safe mean denominator:

    float(np.mean(np.abs(values))) + 1e-8


If selecting negative returns:

    negative_returns = returns[returns < 0]


you MUST handle the case where:

    len(negative_returns) == 0


Do not calculate the mean of an empty array.

Do not calculate the standard deviation of an empty array.


============================================================
7. ALLOWED FEATURE CONCEPTS
============================================================

The four features may represent general concepts such as:

    short-term momentum

    medium-term momentum

    trend strength

    realized volatility

    downside risk

    mean-reversion strength

    current price deviation from moving average

    high-low range

    return dispersion

    normalized volume

    price-volume interaction


These are feature CONCEPTS.

Implement them directly using NumPy.


Do NOT generate:

    ticker-specific features

    company-specific features

    sector-specific features

    portfolio correlation

    cross-sectional ranking

    covariance matrices

    portfolio beta

    diversification score


Those require information from multiple assets and do not
belong inside revise_state(s).


============================================================
8. REQUIRED PROGRAM SHAPE
============================================================

Keep the generated program structurally similar to:

    import numpy as np

    def revise_state(s):
        x = np.asarray(s, dtype=np.float64)
        closes = x[0::6]
        highs = x[2::6]
        lows = x[3::6]
        volumes = x[4::6]
        returns = np.diff(closes) / (np.abs(closes[:-1]) + 1e-8)
        f1 = VALID_SIMPLE_NUMPY_SCALAR
        f2 = VALID_SIMPLE_NUMPY_SCALAR
        f3 = VALID_SIMPLE_NUMPY_SCALAR
        f4 = VALID_SIMPLE_NUMPY_SCALAR
        extra = np.array([f1, f2, f3, f4], dtype=np.float64)
        extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
        return np.concatenate((x, extra))

    def intrinsic_reward(updated_s):
        f1 = float(updated_s[120])
        f2 = float(updated_s[121])
        f3 = float(updated_s[122])
        f4 = float(updated_s[123])
        phi = VALID_SIMPLE_SCALAR_EXPRESSION
        phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
        return phi


Replace:

    VALID_SIMPLE_NUMPY_SCALAR

and:

    VALID_SIMPLE_SCALAR_EXPRESSION


with real executable expressions.

Do NOT output those placeholder words.


============================================================
9. SIMPLICITY REQUIREMENT
============================================================

Do not create complicated mathematical expressions just to
appear sophisticated.

Prefer simple and stable formulas.

BAD:

    multiple nested np.where calls

BAD:

    deeply nested parentheses

BAD:

    large list comprehensions

BAD:

    nested lists

BAD:

    large matrices

BAD:

    many temporary variables


GOOD:

    simple momentum

GOOD:

    simple volatility

GOOD:

    simple range

GOOD:

    simple price deviation

GOOD:

    simple downside statistic


============================================================
10. POTENTIAL-BASED INTRINSIC REWARD SEMANTICS
============================================================

The function name intrinsic_reward is retained only for API
compatibility. It does NOT directly produce the final PPO reward.

Its output is a bounded state potential Phi(s). Phi(s) measures
how desirable the CURRENT financial state is using f1-f4.

Examples of valid directional meaning:

    stronger positive momentum -> higher potential
    higher downside risk       -> lower potential
    higher volatility          -> lower potential

The environment constructs the shaping reward as:

    F(s_t, s_next) = gamma * Phi(s_next) - Phi(s_t)

Therefore, DO NOT encode any of the following in Phi(s):

    realized portfolio return
    transaction cost
    portfolio profit or loss
    the final PPO reward
    a bonus for the current action

Use a simple signed weighted combination of f1-f4. A good state
and a bad state must be distinguishable; do not use abs() in a
way that rewards equally large positive and negative signals.

Phi(s) must remain bounded. Return a value in [-1.0, 1.0],
preferably using:

    np.tanh(...)


Example conceptual form:

    phi = np.tanh(0.5 * f1 - 0.5 * abs(f3))


Do not copy this blindly.

Adapt the formula to the selected four feature meanings and
return float(phi).


============================================================
11. SYNTAX VALIDATION
============================================================

Before returning the program, internally verify:

    compile(code, "<generated>", "exec")

would succeed.


Check:

    every '(' has ')'

    every '[' has ']'

    every quote is closed

    every function has a return statement

    indentation is valid

    every variable is defined before use


Pay special attention to the last 10 lines.

Do not stop generation before intrinsic_reward is completely
finished.


============================================================
12. FINAL RESPONSE FORMAT
============================================================

Return ONLY executable Python source code.

Do NOT output markdown.

Do NOT output:

    ```python

Do NOT output:

    ```

Do NOT explain the code.

Do NOT provide reasoning.

Do NOT output prose before:

    import numpy as np

Do NOT output prose after:

    return phi
"""


DIRECT_REWARD_SEMANTICS = """
============================================================
10. DIRECT INTRINSIC REWARD SEMANTICS (PURE GIFT CONTROL)
============================================================

The function intrinsic_reward(updated_s) directly produces the
bounded LLM-generated auxiliary reward that the environment adds to
the portfolio reward at the current transition.

Use f1-f4 to reward financially desirable signals and penalize
undesirable signals. Examples:

    stronger positive momentum -> higher intrinsic reward
    higher downside risk       -> lower intrinsic reward
    higher volatility          -> lower intrinsic reward

This is the original/direct GIFT control condition. The environment
does NOT compute gamma*Phi(s_next)-Phi(s_current) in this condition.

The direct intrinsic reward must remain finite and bounded in
[-1.0, 1.0], preferably using np.tanh(...).

Use a simple signed weighted combination of f1-f4 and return it as a
float. Do not reference portfolio variables that are unavailable in
updated_s.
"""


def _strict_code_contract(pbir_enabled: bool) -> str:
    """Return a non-contradictory generated-code contract for one method."""
    if pbir_enabled:
        return STRICT_CODE_CONTRACT

    contract = STRICT_CODE_CONTRACT
    section_start = contract.index(
        "============================================================\n"
        "10. POTENTIAL-BASED INTRINSIC REWARD SEMANTICS")
    section_end = contract.index(
        "============================================================\n"
        "11. SYNTAX VALIDATION",
        section_start,
    )
    contract = contract[:section_start] + DIRECT_REWARD_SEMANTICS + "\n\n" + contract[section_end:]
    contract = contract.replace("        phi = ...", "        intrinsic_r = ...")
    contract = contract.replace("                phi,", "                intrinsic_r,")
    contract = contract.replace("        return phi", "        return intrinsic_r")
    contract = contract.replace("    phi\n", "    intrinsic_r\n")
    contract = contract.replace("    return phi\n", "    return intrinsic_r\n")
    contract = contract.replace("    return phi", "    return intrinsic_r")
    return contract


# ============================================================
# Reward rule definitions
# ============================================================

REWARD_RULES = {

    "penalize_concentration":
        "Penalty when one risky-asset weight exceeds the configured maximum.",

    "reward_diversification":
        "Bonus for maintaining diversified allocation across available assets.",

    "penalize_turnover":
        "Penalty for excessive portfolio turnover.",

    "regime_defensive":
        "Reward a more defensive portfolio allocation during high-risk regimes.",

    "momentum_alignment":
        "Reward allocation that is consistent with available momentum information.",

    "volatility_scaling":
        "Scale reward downward during high-volatility conditions.",

    "drawdown_penalty":
        "Penalty when portfolio drawdown exceeds the configured threshold.",
}


# ============================================================
# Initial generation prompt
# ============================================================

def build_init_prompt(market_stats: str, pbir_enabled: bool = True) -> str:
    """
    First GIFT iteration.

    Generates:
    - revise_state
    - intrinsic_reward
    """

    reward_objective = (
        "a bounded state potential Phi(s). The environment, not generated "
        "code, turns Phi(s) into the final potential-based shaping reward."
        if pbir_enabled else
        "a bounded direct intrinsic reward. The environment adds this "
        "LLM-generated auxiliary reward directly to the portfolio reward."
    )
    code_contract = _strict_code_contract(pbir_enabled)

    return f"""
You are generating executable Python code for GIFT,
an LLM-guided PPO portfolio optimization framework.


{DATASET_AGNOSTIC_DESC}


{STATE_LAYOUT_DESC}


============================================================
OBJECTIVE
============================================================

Create an improved per-asset state representation and
{reward_objective}

The implementation must generalize across different stock
datasets without modification.

Generate EXACTLY FOUR useful financial features.

Use only the supplied numerical 20-day history.

Possible general goals are:

- capture useful trend information
- measure risk
- detect mean reversion
- measure recent volatility
- detect unusual price range
- detect unusual volume


Do NOT use asset identity.

Do NOT use ticker identity.

Do NOT use company identity.


{code_contract}


============================================================
CURRENT DATASET MARKET STATISTICS
============================================================

The information below was calculated automatically using the
CURRENT training dataset.

A different experimental dataset will produce different market
statistics.

Use this information only to decide which GENERAL financial
features are appropriate.

If asset names appear below, DO NOT hard-code those names.

{market_stats}


============================================================
FINAL TASK
============================================================

Generate one complete executable Python program.

Final checklist:

- no ticker names
- NumPy only
- exactly two functions
- exactly four added features
- revised state length exactly 124
- intrinsic_reward reads generated features from indices 120-123
- no cross-function undefined variables
- finite numerical values
- short code
- valid syntax
- Python source only

Generate now.
"""


# ============================================================
# COT feedback
# ============================================================

def build_cot_prompt(
        sample_results_text: str,
        market_period_summary: str = ""
) -> str:
    """
    Analyze previous GIFT iteration.

    This prompt generates suggestions, NOT Python code.
    """

    return f"""
You are analyzing one completed iteration of a
dataset-independent GIFT portfolio optimization experiment.

THIS STAGE IS ANALYSIS ONLY.

Do NOT output Python code.


============================================================
NEXT-ITERATION IMPLEMENTATION CONSTRAINTS
============================================================

The next generated Python program will use:

- NumPy only
- exactly four additional features
- exactly two functions
- no feature_library imports
- no helper functions
- no ticker-specific behavior
- no portfolio-level calculations inside revise_state
- revised-state dimensions exactly 124

intrinsic_reward will receive generated feature values through:

    updated_s[120]
    updated_s[121]
    updated_s[122]
    updated_s[123]


============================================================
PREVIOUS ITERATION RESULTS
============================================================

{sample_results_text}


============================================================
CURRENT MARKET-PERIOD INFORMATION
============================================================

{market_period_summary}


============================================================
ANALYSIS OBJECTIVES
============================================================

Analyze the previous results and provide concise suggestions
for the next iteration.

Specifically determine:

1. Which generated feature concepts have useful IC.

2. Which generated feature concepts have weak IC.

3. Which generated feature concepts appear useful according
   to SHAP.

4. Which features appear unstable across market conditions.

5. Whether portfolio performance suggests excessive risk.

6. Whether drawdown appears excessive.

7. Whether the state potential has excessive magnitude or variance.

8. Whether the state potential appears poorly aligned with
   actual portfolio performance.

9. Which EXACTLY FOUR general feature concepts should be used
   in the next iteration.

10. Whether Phi(s) should emphasize:
    - momentum,
    - volatility control,
    - downside risk,
    - mean reversion,
    - or a balanced combination.


============================================================
DATASET-INDEPENDENCE REQUIREMENT
============================================================

Different experiments use different asset groups.

Therefore recommendations must remain asset-agnostic.

Even if the results contain ticker names:

Do NOT recommend ticker-specific behavior.


GOOD:

"Replace the weakest volume feature with medium-term momentum."


GOOD:

"Increase the penalty associated with realized volatility."


GOOD:

"Keep the price-deviation feature because its IC is stable."


BAD:

"Give TSLA a larger momentum weight."


BAD:

"Use a special rule for MSFT."


BAD:

"Treat technology stocks differently."


============================================================
IMPLEMENTATION-SAFETY REQUIREMENT
============================================================

Do NOT invent Python function names.

Do NOT recommend feature_library functions.

Do NOT recommend:

    compute_correlation_diversification_score

or any other external helper.

Do NOT recommend more than four generated features.

Do NOT recommend cross-sectional correlation calculations
inside revise_state.

Do NOT provide Python source code.

Only describe simple financial feature concepts that can be
implemented directly with NumPy over one asset's 20-day history.

Keep the response concise.
"""


# ============================================================
# Subsequent generation prompt
# ============================================================

def build_next_iteration_prompt(
        market_stats: str,
        history_text: str,
        cot_suggestions: str = "",
        pbir_enabled: bool = True,
) -> str:
    """
    Iterations 2+.

    Generates revised NumPy-only code based on previous
    diagnostics.
    """

    code_contract = _strict_code_contract(pbir_enabled)
    reward_formula = (
        "state-potential formula Phi(s)"
        if pbir_enabled else "direct intrinsic-reward formula r_LLM(s)"
    )

    return f"""
You are generating the NEXT executable Python candidate for
GIFT, an LLM-guided PPO portfolio optimization framework.

The exact experimental assets may be different from assets used
in previous runs.

The generated Python implementation MUST remain completely
dataset-independent.


{DATASET_AGNOSTIC_DESC}


{STATE_LAYOUT_DESC}


============================================================
PREVIOUS ITERATION HISTORY
============================================================

{history_text}


============================================================
DIAGNOSTIC FEEDBACK
============================================================

{cot_suggestions}


IMPORTANT:

The diagnostic feedback is ADVISORY ONLY.

The strict code contract below has higher priority.

If diagnostic feedback mentions:

- a ticker name
- a company name
- a nonexistent helper function
- feature_library
- correlation between different assets
- portfolio beta
- diversification score
- more than four features
- structurally complicated Python code

IGNORE that implementation suggestion.

Keep only the useful general financial concept and implement
that concept using a simple NumPy expression.


============================================================
CURRENT DATASET MARKET STATISTICS
============================================================

{market_stats}


The statistics above describe only the CURRENT dataset.

Another experiment may use completely different assets.

Never hard-code asset names appearing in these statistics.


{code_contract}


============================================================
ITERATION IMPROVEMENT RULE
============================================================

The structural form of the Python program should remain nearly
unchanged between iterations.

Improve primarily:

    f1 formula

    f2 formula

    f3 formula

    f4 formula

and:

    {reward_formula}


Do NOT increase source-code complexity simply because this is
a later iteration.

Do NOT add a fifth feature.

Do NOT add helper functions.

Do NOT add imports.

Do NOT use variables created in revise_state directly inside
intrinsic_reward.


============================================================
CRITICAL SCOPE CHECK
============================================================

Before outputting intrinsic_reward, verify that each variable
used by intrinsic_reward was created inside intrinsic_reward.

The safest required beginning is:

    def intrinsic_reward(updated_s):
        f1 = float(updated_s[120])
        f2 = float(updated_s[121])
        f3 = float(updated_s[122])
        f4 = float(updated_s[123])


Then use only:

    f1
    f2
    f3
    f4
    phi


Do NOT use an undefined name such as:

    mr

    momentum

    mom

    vol

    volatility

    downside

    zscore

    signal


unless that variable was explicitly assigned inside
intrinsic_reward.


============================================================
FINAL VALIDATION
============================================================

Before answering verify:

1. no ticker symbol is hard-coded

2. import numpy as np exists

3. exactly two functions exist

4. exactly four generated features exist

5. revised state contains exactly 124 dimensions

6. intrinsic_reward uses only locally defined variables

7. no feature_library import exists

8. no external function is referenced

9. all parentheses are balanced

10. all square brackets are balanced

11. every function returns

12. all generated values are finite

13. output code is short

14. output would compile with Python


============================================================
FINAL TASK
============================================================

Return ONLY the complete executable Python source code.

No markdown.

No explanation.

No reasoning.
"""


# ============================================================
# Reward configuration prompt
# ============================================================

def build_reward_config_prompt(
        market_stats: str,
        iteration: int,
        history: list = None,
        feature_rationale: str = "",
        strategy_hint: str = ""
) -> str:
    """
    Ask LLM to select reward rules.

    This remains JSON-based because gift_controller parses the
    response with _extract_json().
    """

    rules_cat = "\n".join(
        f"- {name}: {description}"
        for name, description in REWARD_RULES.items()
    )

    history_text = ""

    if history:

        history_lines = []

        for item in history[-3:]:

            history_lines.append(
                "Iteration "
                + str(item.get("iteration", "?"))
                + ": Sharpe="
                + str(item.get("sharpe", "N/A"))
                + ", MDD="
                + str(item.get("max_drawdown", "N/A"))
                + ", Return="
                + str(item.get("total_return", "N/A"))
            )

        history_text = "\n".join(history_lines)

    return f"""
You are configuring additional reward rules for a
dataset-independent PPO portfolio optimizer.

Different experiments may contain different financial assets.

Do NOT assume a fixed ticker list.

Do NOT use company-specific reasoning.


============================================================
ITERATION
============================================================

{iteration}


============================================================
AVAILABLE REWARD RULES
============================================================

{rules_cat}


============================================================
CURRENT DATASET MARKET STATISTICS
============================================================

{market_stats}


============================================================
GENERATED FEATURE RATIONALE
============================================================

{feature_rationale}


============================================================
CURRENT STRATEGY GUIDANCE
============================================================

{strategy_hint}


============================================================
PREVIOUS ITERATION RESULTS
============================================================

{history_text}


============================================================
TASK
============================================================

Select between 2 and 4 reward rules.

Use ONLY exact rule names listed under AVAILABLE REWARD RULES.

Do NOT invent new reward-rule names.

The selected rules should improve GENERAL portfolio behavior.

Possible objectives include:

- reduce excessive drawdown
- reduce excessive turnover
- reduce concentration
- encourage useful diversification
- handle high-volatility regimes
- align allocations with general momentum information


Do NOT select rules because of the identity of a specific
company.

Set:

    lambda

between:

    0.1

and:

    1.0


Higher lambda represents stronger risk aversion.


============================================================
STRICT JSON OUTPUT CONTRACT
============================================================

Return exactly ONE valid JSON object.

Do NOT return markdown.

Do NOT return:

    ```json

Do NOT return comments.

Do NOT return explanatory text before JSON.

Do NOT return explanatory text after JSON.


Required schema:

{{
  "reward_rules": [
    {{
      "rule": "penalize_turnover",
      "params": {{
        "threshold": 0.1,
        "penalty": 0.15
      }}
    }},
    {{
      "rule": "drawdown_penalty",
      "params": {{
        "dd_threshold": 0.1,
        "penalty": 0.15
      }}
    }}
  ],
  "lambda": 0.5,
  "rationale": "brief dataset-independent explanation"
}}


Return JSON only.
"""
