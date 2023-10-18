from celery import shared_task
import time
from .const import *
from .utils import preprocess_dir, TriggerWordDataset, train_model, get_model
from tensorflow import keras
from tensorflow.data import Dataset
import shutil
import datetime 
from django.urls import reverse
from django.core.mail import send_mail
import logging
import tensorflow as tf
from django.conf import settings
logging.basicConfig(format=f"%(pathname)s - %(lineno)s - %(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s)",filename="logs.txt",filemode="a",level=logging.INFO)

LOGGER = logging.getLogger(__name__)
# preprocess_dir(r"activates_helloworld_mp3","activates_processed",conversion=True,sr_correct=True, trim_pad=False,
#                change_loudness= CONTROL_ACTIVATES_LOUDNESS,  min_mean=ACTIVATE_LOUDNESS_RANGE[0], max_mean=ACTIVATE_LOUDNESS_RANGE[1],
#                seconds=10,pad_method="repeat",add_suffix=False,sr=44100)
from .models import TrainingJob

def send_completion_mail(email, status, url, name):
    if status == "Completed":
        subject = "Training Job has finished"
    if status == "Failed":
        subject = "Training Job Failed"
    from_email = settings.EMAIL_HOST_USER
    to_email = [email]
    if status == "Completed":
        message = f"You training job with the name {name} has finished. Find more details in {url}"
    else:
        message = f"You training job with the name {name} has failed. Find more details in {url}"

        
    try:
        send_mail(subject=subject, message=message, from_email=from_email, recipient_list=to_email, fail_silently=False)
    except Exception as exc:
        LOGGER.info(f" Unable to send mail due to some error {exc}")
        raise type(exc)
    

@shared_task(bind = True)
def test_func(self, tj, epochs, url):
    print(os.getcwd())
    # print("came here")
    # print("hellop")
    os.makedirs("static/progress", exist_ok=True)
    os.makedirs("static/models", exist_ok=True)
    os.makedirs("static/datasets", exist_ok=True)

    tj = TrainingJob.objects.get(id =tj)
    file_name = "static/progress/" + tj.file_name
    with open(file_name, "a") as f:
        pass    
    try:
        proj = tj.project
        data = proj.activatedata_set.all()
        with open(file_name, "a") as f:
            f.writelines("Reading files ...")
        temp_dir_name = f"static/{proj.id}_temp_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        try:        
            os.makedirs(temp_dir_name, exist_ok=False)
        except:
            pass

        for i in data:
            shutil.copy(i.file.name, temp_dir_name)
            # print(i.file.name.split("/")[-1])
        with open(file_name, "a") as f:
            f.writelines("<strong>Completed</strong>")
            f.writelines("\n")
            f.writelines("Preprocessing Data ...")
            
        preprocess_dir(temp_dir_name,f"{temp_dir_name}_processed",conversion=True,sr_correct=True, trim_pad=False,
                change_loudness= CONTROL_ACTIVATES_LOUDNESS,  min_mean=ACTIVATE_LOUDNESS_RANGE[0], max_mean=ACTIVATE_LOUDNESS_RANGE[1],
                seconds=10,pad_method="repeat",add_suffix=False,sr=44100)
        shutil.rmtree(temp_dir_name)
        with open(file_name, "a") as f:
            f.writelines("<strong>Completed</strong>")
            f.writelines("\n")
            f.writelines("Creating Training Data ...")
            
        trigger_word_dataset = TriggerWordDataset(data_dir="",ty = 1375, output_shape = (1375, 1), spec_params=SPEC_PARAMS,
                                            background_length= BACKGROUND_LENGTH ,sr=  SR , backgrounds_dir=r"static\backgrounds_processed", 
                                            change_background_loudness=ADJUST_BACKGROUND_LOUNDESS, change_activate_loudeness=ADJUST_ACTIVATES_LOUDNESS, 
                                            change_negative_loudness= ADJUST_NEGATIVES_LOUDNESS,
                                            negatives_dir=r"static\negatives_processed", activates_dir=f"{temp_dir_name}_processed", demo_data_save_perc=SAVE_DATA_PERCENTAGE)
        

        
        tf_dataset = Dataset.from_generator(trigger_word_dataset.create_dataset_generator,
                                                    output_signature= (tf.TensorSpec(shape=(5511, 101), dtype=tf.float32), tf.TensorSpec(shape=(1375, 1), dtype=tf.float32)),
                                                    args = [N_SAMPLES])
        with open(file_name, "a") as f:
            f.writelines("<strong>Completed</strong>")
            f.writelines("\n")
            f.writelines("Saving Data ...")

            
        tf_dataset.save(os.path.join(r"static\datasets", tj.file_name[:-4]))
        with open(file_name, "a") as f:
            f.writelines("<strong>Completed</strong>")
            f.writelines("\n")
            f.writelines("</h2>")
            f.writelines("""<h2 style="font-weight: bolder">Training Logs : </h2>""")
            f.writelines("<div>")
            f.writelines('<table style="background-color: white; width: 500px"" class="table">')
            f.writelines('''<tr><th background-color: lightskyblue;>Epoch</th><th background-color: lightskyblue;>Accuracy</th><th background-color: lightskyblue;`>Loss</th></tr></table>''')
        shutil.rmtree(f"{temp_dir_name}_processed")
        MODEL_SAVE_DIR = os.path.join(r"static/models", tj.file_name[:-4])
        class CustomCallback(keras.callbacks.Callback):
            def on_epoch_end(self,epoch, logs=None):
                with open(file_name, "r") as f:
                    text = f.read() 
                text = text[:-8]
                with open(file_name, "w") as f:
                    acc = logs.get("accuracy")
                    loss = logs.get("loss")
                    text += f"<td>{acc}</td>"
                    text += f"<td>{loss}</td>" + "</tr>" + "</table>"
                    f.write(text)
                    # acc = logs.get("accuracy")
                    # loss = logs.get("loss")
                    # f.writelines(f"<td>{acc}</td>")
                    # f.writelines(f"<td>{loss}</td>")
                    # f.writelines("</tr>")
                    # f.writelines("\n")  
            def on_epoch_begin(self,epoch, logs=None):
                with open(file_name, "r") as f:
                    text = f.read() 
                text = text[:-8]
                with open(file_name, "w") as f:
                    text += "<tr>"
                    text += f"<td>{epoch}</td>" + "</table>"
                    f.write(text)
                    # f.writelines("<tr>")
                    # f.writelines(f"<td>{epoch}</td>")
                    # s
                    # f.writelines("\n")
            def on_train_end(self, logs=None):
                with open(file_name, "a") as f:
                    
                    f.writelines("</table>")
                    f.writelines("</div>")
                    # f.writelines("\n")
            
        CALLBACKS = [
        CSVLogger(filename=os.path.join(MODEL_SAVE_DIR, "training_logs.csv"), append=True),
        ModelCheckpoint(filepath=os.path.join(MODEL_SAVE_DIR, "checkpoint_{epoch}_{loss}.h5"), verbose=True,save_freq="epoch"),
        CustomCallback(),
    ]

        train_model(get_model(),os.path.join(r"static\datasets", tj.file_name[:-4]) ,"",epochs,CALLBACKS,32,metrics= ["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()], optimizer= "adam", loss= "binary_crossentropy", save_path=MODEL_SAVE_DIR, 
              class_weight=CLASS_WEIGHT)
        shutil.rmtree(os.path.join(r"static\datasets", tj.file_name[:-4]))

        with open(file_name, "a") as f:
            f.writelines("<strong><h1>Training Job Finished</h1></strong>")
            f.writelines("\n")

        tj.status = "Completed"
        
        send_completion_mail(proj.user.email, "Completed", url, tj.name)
        tj.save()
    except Exception as exc:
        LOGGER.info(f"Error {exc}")
        send_completion_mail(proj.user.email, "Failed", url, tj.name)
        tj.status = "Failed"
        tj.save()
