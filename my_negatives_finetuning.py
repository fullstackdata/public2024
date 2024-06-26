import json
import random
train_path = "negative.json" # from https://github.com/urchade/GLiNER/blob/main/examples/sample_data.json

with open(train_path, "r") as f:
    data = json.load(f)

print('Dataset size:', len(data))
#shuffle
random.shuffle(data)    
print('Dataset is shuffled...')

train_data = data[:int(len(data)*0.9)]
test_data = data[int(len(data)*0.9):]

print('Dataset is splitted...')


import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import torch
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset

    
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


model = GLiNER.from_pretrained("urchade/gliner_small")

###########################################

train_dataset = GLiNERDataset(train_data, model.config, data_processor=model.data_processor)
test_dataset = GLiNERDataset(test_data, model.config, data_processor=model.data_processor)

data_collator = DataCollatorWithPadding(model.config)


#Optional: compile model for faster trainingtorch.set_float32_matmul_precision('high')model.to(device)model.compile_for_training()
###########################################
training_args = TrainingArguments(
    output_dir="models",
    learning_rate=5e-6,
    weight_decay=0.01,
    others_lr=1e-5,
    others_weight_decay=0.01,
    lr_scheduler_type="linear", #cosine
    warmup_ratio=0.1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_steps = 1000,
    save_total_limit=10,
    dataloader_num_workers = 1,
    use_cpu = False,
    report_to="none",
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=model.data_processor.transformer_tokenizer,
    data_collator=data_collator,
    
)
trainer.train()
###########################################
text = """
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
"""
email_text = """Sarah clicked on a phishing email."""
em2 = "She glanced at her overflowing inbox with a sigh. Dozens of unread emails sat there, one email is from India"

em3 = """she has a mix of work emails, a personal email and nagging reminders from her bank. 
She knew she should tackle them, but she has to plan her Italy trip first. """

em4 = """Sarah has to plan her Italy trip. With a determined breath, she decided to start with the most important email and hoped to clear out the rest by the end of the day."""

# Labels for entity prediction
labels = ["Person", "Communincation", "Country"] # for v2.1 use capital case for better performance

# Perform entity prediction
entities = model.predict_entities(em4, labels, threshold=0.7)

# Display predicted entities and their labels
for entity in entities:
    print(entity["text"], "=>", entity["label"])
