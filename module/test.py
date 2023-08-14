import torch, math, time, evaluate



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader

        self.task = config.task
        self.device = config.device
        self.model_type = config.model_type
        
        self.metric_name = 'BLEU' if self.task == 'nmt' else 'ROUGE'
        self.metric_module = evaluate.load(self.metric_name.lower())
        


    def test(self):
        score = 0.0         
        self.model.eval()

        with torch.no_grad():
            for batch in self.dataloader:

                src = batch['src'].to(self.device)
                label = self.tokenizer.decode(batch['trg'].tolist())            
        
                pred = self.model.generate(src)
                score += self.metric_score(pred, label)

        txt = f"TEST Result on {self.task.upper()} with {self.model_type.upper()} model"
        txt += f"\n-- Score: {round(score/len(self.dataloader), 2)}\n"
        print(txt)


    def metric_score(self, pred, label):
        if not pred:
            return 0.00

        #For Translation and Summarization Tasks
        if self.task == 'nmt':
            score = self.metric_module.compute(
                predictions=[pred], 
                references =[[label]]
            )['bleu']

        else:
            score = self.metric_module.compute(
                predictions=[pred], 
                references =[[label]]
            )['rouge2']

        return score * 100