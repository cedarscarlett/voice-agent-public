import re
from pathlib import Path

def normalize_timestamps(log_text: str) -> str:
    """
    Normalize all 't_ns' values in the log:
    - subtract the first t_ns found
    - divide by 1e9 (nanoseconds -> seconds)
    """

    pattern = re.compile(r"'t_ns':\s*(\d+)")
    matches = list(pattern.finditer(log_text))

    if not matches:
        return log_text  # no timestamps found

    t0 = int(matches[0].group(1))

    def replacer(match: re.Match) -> str:
        t_ns = int(match.group(1))
        t_sec = (t_ns - t0) / 1e9
        return f"'t_ns': {t_sec:.9f}"

    return pattern.sub(replacer, log_text)


if __name__ == "__main__":
    raw_log = """5187:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271738611631}
5194:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271740311506}
5199:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271740578204}
5204:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271741010978}
5209:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271741239824}
5214:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271741867924}
5219:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271742085733}
5224:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271742287770}
5229:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271742489178}
5234:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271745598917}
5239:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271745908138}
5244:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271746162566}
5255:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271747304012}
5260:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271747551240}
5265:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271748318112}
5270:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271748732697}
5275:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271749124849}
5280:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271750200289}
5285:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271750722549}
5290:REDUCER ENTRY {'event': <EventType.LLM_TOKEN: 'LLM_TOKEN'>, 't_ns': 2271751433605}
5297:REDUCER ENTRY {'event': <EventType.LLM_DONE: 'LLM_DONE'>, 't_ns': 2271752733510}"""

    normalized = normalize_timestamps(raw_log)

    out_path = Path(__file__).parent / "temp_log.txt"
    out_path.write_text(normalized, encoding="utf-8")
