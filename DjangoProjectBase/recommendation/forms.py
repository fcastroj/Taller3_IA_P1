from django import forms

class RecommendationForm(forms.Form):
    prompt = forms.CharField(label="Describe la película que buscas", max_length=200)
 