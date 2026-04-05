import json
import os
import re


START_MARKER = "{#- chem_llm_default_system_prompt_start -#}"
END_MARKER = "{#- chem_llm_default_system_prompt_end -#}"

# References:
# https://huggingface.co/docs/transformers/en/chat_templating_writing
# https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/llama-2-chat.jinja


def marker_block(system_prompt):
    quoted = json.dumps(system_prompt)
    return (
        f"    {START_MARKER}\n"
        f"    {{%- set system_message = {quoted} %}}\n"
        f"    {END_MARKER}"
    )


def insert_default_system_prompt(chat_template, system_prompt):
    """Insert or replace a system prompt block inside a Jinja chat template."""
    prompt = (system_prompt or "").strip()
    if not prompt:
        return chat_template

    block = marker_block(prompt)
    if START_MARKER in chat_template and END_MARKER in chat_template:
        pattern = re.compile(
            rf"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}",
            flags=re.DOTALL,
        )
        return pattern.sub(lambda x: block.strip(), chat_template, count=1)

    # Fallback: replace the empty system_message assignment
    target = '{%- set system_message = "" %}'
    if target in chat_template:
        return chat_template.replace(target, block.strip(), 1)

    return chat_template


def patch_text_file(path, system_prompt):
    """Patch a plain-text Jinja template file in-place."""
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        original = f.read()
    updated = insert_default_system_prompt(original, system_prompt)
    if updated == original:
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)
    return True


def patch_tokenizer_config(tokenizer_config_path, system_prompt):
    """Patch the chat_template field inside a tokenizer_config.json."""
    if not os.path.exists(tokenizer_config_path):
        return False

    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chat_template = data.get("chat_template")
    if not isinstance(chat_template, str):
        return False

    updated = insert_default_system_prompt(chat_template, system_prompt)
    if updated == chat_template:
        return False

    data["chat_template"] = updated
    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return True


def patch_chat_template_jinja(chat_template_path, system_prompt):
    """Patch a standalone .jinja chat template file."""
    return patch_text_file(chat_template_path, system_prompt)