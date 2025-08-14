import streamlit as st
from utils.helpers import segment_clauses, simplify_clause

def clause_simplification_ui(text):
    st.header("Clause Simplification")
    clauses = segment_clauses(text)
    for i, clause in enumerate(clauses):
        simplified = simplify_clause(clause)
        st.markdown(f"**Clause {i+1}:**")
        st.write(f"**Original:** {clause}")
        st.write(f"**Simplified:** {simplified}")

def simplify_clause(clause):
    conv = [{
        "role": "user",
        "content": (
            "Rewrite the following legal clause in plain English for someone with no legal background. "
            "Use short sentences, simple words, and remove any unnecessary legal jargon. "
            "If the clause is already simple, rephrase it to be even clearer:\n\n"
            f"{clause}"
        )
    }]
    input_ids = tokenizer.apply_chat_template(conv, return_tensors="pt", return_dict=True, add_generation_prompt=True).to(DEVICE)
    set_seed(42)
    output = model.generate(
        **input_ids,
        max_new_tokens=512,
    )
    prediction = tokenizer.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return prediction.strip()