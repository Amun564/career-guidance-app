#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('career_model.pkl')

st.title("Career Guidance System")
st.markdown("### Rate your skills (1-20 scale)")

# Intelligence sliders
scores = {}
for col in model.feature_names_in_:
    scores[col] = st.slider(
        col.replace('_', ' ').title(),
        1, 20, 10,
        help=f"Your {col.replace('_', ' ')} intelligence score"
    )

if st.button("Get Career Recommendation"):
    # Prediction
    input_df = pd.DataFrame([scores])
    prediction = model.predict(input_df)[0]
    
    # Display results
    st.success(f"## Recommended Career: {prediction}")
    
    # Career description (placeholder - add real descriptions)
    st.markdown(f"**About {prediction}:**\n\n[Career description would go here]")
    
    # Top 3 similar careers
    probas = model.predict_proba(input_df)[0]
    top_3 = sorted(zip(model.classes_, probas), key=lambda x: -x[1])[:3]
    
    st.markdown("### You might also consider:")
    for career, prob in top_3[1:]:  # Skip first (predicted career)
        st.markdown(f"- {career} ({prob:.1%} match)")

    # Feature importance visualization
    st.markdown("### What influenced this recommendation:")
    importances = pd.DataFrame({
        'Skill': model.feature_names_in_,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    st.bar_chart(importances.set_index('Skill'))


# In[ ]:




