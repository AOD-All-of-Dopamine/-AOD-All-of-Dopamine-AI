def write_eyeball_dump(records, out_path: str) -> None:
    lines: list[str] = ["# M1 fun_tag 눈검수 덤프", ""]
    for target, extraction in records:
        lines.append(f"## [{target.content_id}] {target.master_title}")
        lines.append(f"- 장르: {', '.join(target.genres)}")
        lines.append(f"- normalized_summary: {extraction.normalized_summary}")
        lines.append(f"- extraction_quality: {extraction.extraction_quality:.3f}")
        lines.append("- fun_tags:")
        for item in extraction.fun_tags:
            flag = " (proposed/new)" if item.is_new else ""
            lines.append(
                f"  - {item.tag}{flag} | score={item.tag_score:.2f} "
                f"conf={item.tag_confidence:.2f} | 근거: {item.evidence}"
            )
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
