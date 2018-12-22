docker run -t --rm -p 8501:8501 \
   -v "$(pwd)/model/half_plus_ten:/models/half_plus_ten" \
   -e MODEL_NAME=half_plus_ten \
   tensorflow/serving

docker run -t --rm -p 8501:8501 \
   -v "$(pwd)/pb_model:/models/half_plus_ten" \
   -e MODEL_NAME=half_plus_ten \
   tensorflow/serving

curl -d '{"instances": [1.0, 2.0, 5.0]}' -X POST http://localhost:8501/v1/models/half_plus_ten:predict