import os
import openai
from openai import OpenAI
from typing import Union, List


def get_llm_response(pre_prompt, prompt, temperature=0.0, stop: Union[str, List[str]] = "END", model: str = "gpt-4", seed=0, max_tokens: int = None, want_finish_reason=False) -> str:
    client = OpenAI()
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    openai_response = client.chat.completions.create(
        # model="gpt-4-1106-preview",
        model=model,
        messages=[
            {"role": "system", "content": pre_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        stop=stop,
        seed=seed,
        max_tokens=max_tokens,
    )
    # text = openai_response['choices'][0]['message']['content']
    text = openai_response.choices[0].message.content.strip()
    if want_finish_reason:
        finish_reason = openai_response.choices[0].finish_reason
        return text, finish_reason
    else:
        return text
