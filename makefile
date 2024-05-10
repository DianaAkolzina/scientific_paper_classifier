run_api:
	uvicorn api.app:app --reload --port 8002

run_main:
	python -m interface.main

run_main_api:
	python -m interface.main_api

run_base_pred:
	python -c 'from interface.main_api import predict_baseline; predict_baseline()'
