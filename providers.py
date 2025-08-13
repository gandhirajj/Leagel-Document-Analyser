import os
from typing import List, Dict, Optional


class Providers:
	def __init__(self) -> None:
		# Watson NLU
		self._watson_api_key = os.getenv("IBM_WATSON_APIKEY")
		self._watson_url = os.getenv("IBM_WATSON_URL")
		self._watson_version = os.getenv("IBM_WATSON_VERSION", "2023-10-01")
		self.is_watson_ready = bool(self._watson_api_key and self._watson_url)

		# watsonx.ai
		self._wx_api_key = os.getenv("IBM_WATSONX_APIKEY")
		self._wx_project_id = os.getenv("IBM_WATSONX_PROJECT_ID")
		self._wx_region = os.getenv("IBM_WATSONX_REGION", "us-south")
		self._wx_url = os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
		self._granite_model = os.getenv("IBM_GRANITE_MODEL_ID", "ibm/granite-13b-instruct-v2")
		self.is_watsonx_ready = bool(self._wx_api_key and self._wx_project_id)

		# Local Granite via Hugging Face
		self._hf_token = os.getenv("HF_TOKEN")
		self._hf_model = os.getenv("HF_GRANITE_MODEL", "ibm-granite/granite-3.3-2b-instruct")
		self._hf_device = os.getenv("HF_DEVICE", "cpu")
		self.is_hf_granite_enabled = bool(self._hf_model)

		# Lazy clients
		self._watson_client = None
		self._wx_client = None
		self._hf_model_client = None
		self._hf_tokenizer = None

	def _get_watson_client(self):
		if not self.is_watson_ready:
			return None
		if self._watson_client is not None:
			return self._watson_client
		try:
			from ibm_watson import NaturalLanguageUnderstandingV1
			from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
			authenticator = IAMAuthenticator(self._watson_api_key)
			client = NaturalLanguageUnderstandingV1(version=self._watson_version, authenticator=authenticator)
			client.set_service_url(self._watson_url)
			self._watson_client = client
			return client
		except Exception:
			return None

	def _get_wx_client(self):
		if not self.is_watsonx_ready:
			return None
		if self._wx_client is not None:
			return self._wx_client
		try:
			from ibm_watsonx_ai import Credentials, WatsonxLLM
			creds = Credentials(api_key=self._wx_api_key, url=self._wx_url)
			model = WatsonxLLM(model_id=self._granite_model, credentials=creds, project_id=self._wx_project_id)
			self._wx_client = model
			return model
		except Exception:
			return None

	def _get_hf_granite(self):
		if not self.is_hf_granite_enabled:
			return None, None
		if self._hf_model_client is not None and self._hf_tokenizer is not None:
			return self._hf_model_client, self._hf_tokenizer
		try:
			import torch
			from transformers import AutoModelForCausalLM, AutoTokenizer
			kwargs = {"device_map": self._hf_device}
			if self._hf_device.startswith("cuda"):
				kwargs["torch_dtype"] = torch.bfloat16
			model = AutoModelForCausalLM.from_pretrained(self._hf_model, use_auth_token=self._hf_token, **kwargs)
			tokenizer = AutoTokenizer.from_pretrained(self._hf_model, use_auth_token=self._hf_token)
			self._hf_model_client = model
			self._hf_tokenizer = tokenizer
			return model, tokenizer
		except Exception:
			return None, None

	def watson_nlu_entities(self, text: str) -> List[Dict]:
		client = self._get_watson_client()
		if not client:
			return []
		try:
			from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions
			resp = client.analyze(text=text, features=Features(entities=EntitiesOptions(limit=250, mentions=True))).get_result()
			out: List[Dict] = []
			for ent in resp.get("entities", []):
				out.append({
					"text": ent.get("text"),
					"type": ent.get("type"),
					"start": ent.get("location", [None, None])[0],
					"end": ent.get("location", [None, None])[1],
				})
			return out
		except Exception:
			return []

	def watson_nlu_classify(self, text: str) -> Dict:
		client = self._get_watson_client()
		if not client:
			return {}
		try:
			from ibm_watson.natural_language_understanding_v1 import Features, CategoriesOptions
			resp = client.analyze(text=text, features=Features(categories=CategoriesOptions(limit=3))).get_result()
			cats = resp.get("categories", [])
			if not cats:
				return {}
			best = max(cats, key=lambda c: c.get("score", 0))
			label = best.get("label", "/")
			mapped = "Service Agreement"
			if "employment" in label:
				mapped = "Employment Agreement"
			elif "lease" in label or "real estate" in label:
				mapped = "Lease"
			elif "confidential" in label or "nda" in label:
				mapped = "NDA"
			return {"label": mapped, "score": float(best.get("score", 0.7)), "raw": cats}
		except Exception:
			return {}

	def watsonx_generate(self, prompt: str) -> Optional[str]:
		client = self._get_wx_client()
		if not client:
			# Try local Granite
			model, tok = self._get_hf_granite()
			if model is not None and tok is not None:
				import torch
				from transformers import set_seed
				messages = [{"role": "user", "content": prompt}]
				inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
				set_seed(42)
				with torch.inference_mode():
					output = model.generate(**inputs, max_new_tokens=1024)
				text = tok.decode(output[0, inputs.shape[1]:], skip_special_tokens=True)
				return text
			return None
		try:
			resp = client.generate(prompt)
			# Different SDKs may return different shapes; try to normalize
			if isinstance(resp, dict):
				text = (
					resp.get("results", [{}])[0].get("generated_text")
					or resp.get("completions", [{}])[0].get("text")
					or resp.get("result")
				)
				return text
			return str(resp)
		except Exception:
			return None

	def watsonx_classify_zero_shot(self, text: str, labels: List[str]) -> Optional[str]:
		client = self._get_wx_client()
		if not client:
			# Try local Granite
			model, tok = self._get_hf_granite()
			if model is not None and tok is not None:
				from transformers import set_seed
				prompt = (
					"Classify the following legal document into one of these categories: "
					+ ", ".join(labels)
					+ "\nDocument:\n" + text[:4000] + "\n\nAnswer with only the category name."
				)
				messages = [{"role": "user", "content": prompt}]
				inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
				set_seed(42)
				output = model.generate(**inputs, max_new_tokens=128)
				result = tok.decode(output[0, inputs.shape[1]:], skip_special_tokens=True)
				for l in labels:
					if l.lower() in result.lower():
						return l
				return None
			return None
		try:
			prompt = (
				"Classify the following legal document into one of these categories: "
				+ ", ".join(labels)
				+ "\nDocument:\n" + text[:4000] + "\n\nAnswer with only the category name."
			)
			resp = self.watsonx_generate(prompt)
			return resp.strip() if resp else None
		except Exception:
			return None
