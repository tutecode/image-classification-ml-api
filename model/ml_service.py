import json
import os
import time

import numpy as np
import redis
import settings
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

# TODO
# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
# Connect to Redis
db = redis.Redis(
    host=settings.REDIS_IP,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB_ID,
    # charset="utf-8",
    # decode_responses=True
)

# TODO
# Load your ML model and assign to variable `model`
# See https://drive.google.com/file/d/1ADuBSE4z2ZVIdn66YDSwxKv-58U7WEOn/view?usp=sharing
# for more information about how to use this model.
model = ResNet50(include_top=True, weights="imagenet")
# print(model.summary())


def predict_batch(image_names):
    print("Launching ML service BATCH...")
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    x_batches = []

    for image_name in image_names:
        class_name = None
        pred_probability = None

        # TODO
        # Preprocess the image
        # Get the image from the UPLOAD_FOLDER 
        path_image = settings.UPLOAD_FOLDER + "/" + image_name
        img = image.load_img(path_image, target_size=(224, 224))
        # Convert the PIL image to a Numpy array
        x = image.img_to_array(img)
        # print(f"Size of image is: {x.shape}")

        # Also we must add an extra dimension to this array
        # because our model is expecting as input a batch of images.
        # In this particular case, we will have a batch with a single
        # image inside
        # x_batch = np.expand_dims(x, axis=0)

        # Append the preprocessed image to the batch list
        x_batches.append(x)

    # Convert the batch list to a Numpy array
    nx_batches = np.stack(x_batches, axis=0)
    print(f"batch_image: {nx_batches.shape}")

    # Scale pixel values of the batch
    x_batchs = preprocess_input(nx_batches)

    # Make predictions using the ML model
    preds = model.predict(x_batchs)
    print("number_of_predictions: ", len(preds))

    outputs = []

    for pred in preds:
        class_pred = {}
        # Expand dimensions of the prediction batch array
        batch_pred = np.expand_dims(np.array(pred), axis=0)
        # Decode predictions and extract class name and probability
        res_model = decode_predictions(batch_pred, top=1)[0]
        class_pred["class_name"] = res_model[0][1]
        class_pred["pred_prob"] = round(res_model[0][2], 4)
        outputs.append(class_pred)
        # print(f"class_name: {class_pred['class_name']}, pred: {class_pred['pred_prob']}")

    return outputs


def classify_process_batch():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """

    while True:
        try:
            with db.pipeline() as db_pip:
                # Take a batch of new jobs from Redis with 'lpop'
                msg_list = db.lpop(settings.REDIS_QUEUE, 5)
                if msg_list:
                    print(f"\n########## START BATCH ##########")
                    print(f"Total jobs: {len(msg_list)} \ndata: {msg_list}")

                    # Load JSON data of each job in the batch
                    msgs_loaded = [json.loads(msg) for msg in msg_list]

                    # Extract image names from each job
                    image_names = [json.loads(msg)["image_name"] for msg in msg_list]

                    # Use the loaded ML model to get predictions for the batch of images
                    predictions = predict_batch(image_names)

                    print(
                        f"\nTotal predictions: {len(predictions)} \ndata: {predictions}"
                    )

                    # Store the results in Redis
                    for msg, prediction in zip(msgs_loaded, predictions):
                        # Create JSON data object with predicted class name and score
                        job_data = json.dumps(
                            {
                                "prediction": prediction["class_name"],
                                "score": np.float64(prediction["pred_prob"]),
                            }
                        )

                        # Store the job ID and job_data in Redis
                        db_pip.set(msg["id"], job_data)

                    print("################# END BATCH ######################")

                    # Execute the Redis pipeline to persist the results
                    db_pip.execute()

                    # Sleep for a bit inside of batch
                    time.sleep(settings.SERVER_SLEEP)
        except Exception as exc:
            # Print any exceptions that occur during processing
            print(exc)


# ----------------------------------------------------------------------------------------------------


def predict(image_name):
    """
    Load an image from the corresponding folder based on the image name
    received, then run the ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    class_name = None
    pred_probability = None

    # Load the image
    img_path = os.path.join(settings.UPLOAD_FOLDER, image_name)

    img = image.load_img(img_path, target_size=(224, 224))
    img = img.convert("RGB")  # Convert image to RGB format

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    processed_img = preprocess_input(img_array)
    # print("1", processed_img)

    # Make predictions
    try:
        prediction = model.predict(processed_img)
        # print('2', prediction)
        decoded_predictions = decode_predictions(prediction, top=1)[0]
        class_name = decoded_predictions[0][1]
        pred_probability = round(decoded_predictions[0][2], 4)
    except Exception as e:
        raise SystemExit("ERROR: Failed to make predictions with the model:", str(e))

    return class_name, pred_probability


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        # Inside this loop you should add the code to:
        #   1. Take a new job from Redis
        #   2. Run your ML model on the given data
        #   3. Store model prediction in a dict with the following shape:
        #      {
        #         "prediction": str,
        #         "score": float,
        #      }
        #   4. Store the results on Redis using the original job ID as the key
        #      so the API can match the results it gets to the original job
        #      sent
        # Hint: You should be able to successfully implement the communication
        #       code with Redis making use of functions `brpop()` and `set()`.
        # TODO
        # Take a new job from Redis

        # 1. Take a new job from Redis

        queue_name, msg = db.brpop(settings.REDIS_QUEUE)
        print("msg", msg)

        # 2. Run ML model on the given data
        newmsg = json.loads(msg)

        # 2.1. only need the filename image the image object is loaded by the upload folder
        class_name, pred_probability = predict(newmsg["image_name"])

        # 3. Store model prediction in a dict with the following shape
        res_dict = {
            "prediction": str(class_name),
            "score": np.float64(pred_probability),
        }

        # 4. Store the results on Redis using the original job ID as the key
        # so the API can match the results it gets to the original job sent
        res_id = newmsg["id"]

        # Here, you can see we use `json.dumps` to
        # serialize our dict into a JSON formatted string.
        try:
            db.set(res_id, json.dumps(res_dict))
        except:
            print("ERROR: Results Not Stored")

        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    # classify_process_batch()
    classify_process()
