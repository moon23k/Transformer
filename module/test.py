import torch, math, time, evaluate
from module import Generator
from transformers import BertModel, AutoTokenizer




class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader, test_volumn=100):
        super(Tester, self).__init__()
        
        self.model = model
        self.task = config.task
        self.device = config.device
        self.test_volumn = test_volumn
        self.model_type = config.model_type

        self.tokenizer = tokenizer
        self.dataloader = test_dataloader
        self.generator = Generator(config, model, tokenizer)
        
        if self.task == 'nmt':
            self.metric_name = 'BLEU'
            self.metric_module = evaluate.load('bleu')

        elif self.task == 'dialog':
            mname = "bert-base-uncased"
            self.metric_name = 'BERT'
            self.metric_tokenizer = AutoTokenizer.from_pretrained(mname)
            self.metric_model = BertModel.from_pretrained(mname)
            self.metric_model.eval()

        elif self.task == 'sum':
            self.metric_name = 'ROUGE'
            self.metric_module = evaluate.load('rouge')



    def test(self):
        self.model.eval()
        greedy_score, beam_score = 0.0, 0.0


        for batch in self.dataloader:
        
            src = batch['src'].to(self.device)
            label = batch['trg'].tolist()[0]
    
            greedy_pred = self.generator.generate(src, search='greedy')
            beam_pred = self.generator.generate(src, search='beam')
            
            greedy_score += self.metric_score(greedy_pred, label)
            beam_score += self.metric_score(beam_pred, label)
        
        greedy_score = round(greedy_score / self.test_volumn, 2)
        beam_score = round(beam_score / self.test_volumn, 2)
        
        self.print_rst(greedy_score, beam_score)


    def print_rst(self, greedy_score, beam_score):
        txt = f"TEST Result on {self.task} with {self.model_type} model"
        txt += f"\n-- Greedy Score: {greedy_score}"
        txt += f"\n-- Beam   Score: {beam_score}" 
        print(txt)


    def metric_score(self, pred, label):
        #For Translation and Summarization Tasks
        if self.task != 'dialog':
            
            self.metric_module.add_batch(
                predictions=pred, references=[[l] for l in label]
            )
            
            if self.task == 'nmt':
                score = self.metric_module.compute()['bleu']
            elif self.task == 'sum':        
                score = self.metric_module.compute()['rouge2']

        #For Dialogue Generation Task
        elif self.task == 'dialog':
            
            encoding = self.metric_tokenizer(
                pred, label, padding=True, 
                truncation=True, return_tensors='pt'
            )

            bert_out = self.metric_model(**encoding)[0]

            normalized = torch.nn.functional.normalize(bert_out[:, 0, :], p=2, dim=-1)
            dist = normalized.matmul(normalized.T)
            sim_matrix = dist.new_ones(dist.shape) - dist
            score = sim_matrix[0, 1].item()

        return score * 100