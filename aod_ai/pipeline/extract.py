from aod_ai.models import Extraction


def extract_profile(llm, target, sources, active_tags: list[str]) -> Extraction:
    extraction = llm.extract(metadata=target, sources=sources, active_tags=active_tags)
    active = {t.strip().lower() for t in active_tags}
    corrected = [
        item.model_copy(update={"is_new": item.tag.strip().lower() not in active})
        for item in extraction.fun_tags
    ]
    return extraction.model_copy(update={"fun_tags": corrected})
