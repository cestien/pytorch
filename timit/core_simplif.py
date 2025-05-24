
# Creación de un core simplificado
class Brain:
    
    def __init__(self,modules=None):

        # Set the right device_type
        if self.device == "cpu":
            self.device_type = "cpu"
        elif "cuda" in self.device:
            self.device_type = "cuda"
        else:
            raise ValueError("Expected `self.device` to be `cpu` or `cuda`!")
        # Switch to the right context
        if self.device == "cuda":
            torch.cuda.set_device(0)
        elif "cuda" in self.device:
            torch.cuda.set_device(int(self.device[-1]))
        
        modules = [sb.lobes.models.CRDNN.CRDNN(), sb.nnet.linear.Linear(), 
                   sb.processing.features.InputNormalization(global)]
        self.modules = torch.nn.ModuleDict(modules).to(self.device)

        # Prepare iterating variables
        self.avg_train_loss = 0.0
        self.step = 0
        self.optimizer_step = 0

        # Profiler setup
        self.profiler = None
        if self.profile_training:
            logger.info("Pytorch profiler has been activated.")
            self.tot_prof_steps = (self.profile_steps + self.profile_warmup) - 1
        self.profiler = sb.utils.profiling.prepare_profiling()

    def update_average(self, loss, avg_loss):
        """Update running average of the loss.

        Arguments
        ---------
        loss : torch.tensor
            detached loss, a single float value.
        avg_loss : float
            current running average.

        Returns
        -------
        avg_loss : float
            The average loss.
        """
        if torch.isfinite(loss):
            avg_loss -= avg_loss / self.step
            avg_loss += float(loss) / self.step
        return avg_loss

 
        save_dict = {
            "step": self.step,
            "avg_train_loss": self.avg_train_loss,
            "optimizer_step": self.optimizer_step,
        }
        with open(path, "w", encoding="utf-8") as w:
            w.write(yaml.dump(save_dict))

    def fit(self,epoch_counter,train_set,valid_set=None,):
        if sb.Stage.TRAIN:
            train_set = sb.dataio.dataloader.make_dataloader(train_set, stage=sb.Stage.TRAIN)
        if sb.Stage.VALID:
            valid_set = sb.dataio.dataloader.make_dataloader(valid_set,stage=sb.Stage.VALID)
                
        optimizer = torch.optim.Adadelta(rho=0.95,lr = 'lr',eps=1.e-8)
        # Iterate epochs
        for epoch in epoch_counter:
            # Training stage
            self.on_stage_start(Stage.TRAIN, epoch)
            self.modules.train()
            optimizer.zero_grad()
            for batch in train_set:
                self.step += 1
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
                loss.backward()
                loss.detach().cpu()
                self.optimizer_step += 1
                self.avg_train_loss = self.update_average(loss, self.avg_train_loss)
                self.profiler.step()
            # Run train "on_stage_end" on all processes
            optimizer.zero_grad(set_to_none=True)  # flush gradients
            self.on_stage_end(Stage.TRAIN, self.avg_train_loss, epoch)
            self.avg_train_loss = 0.0
            self.step = 0
            # Validation step
            self.on_stage_start(Stage.VALID, epoch)
            self.modules.eval()
            avg_valid_loss = 0.0
            with torch.no_grad():
                for batch in valid_set:
                    self.step += 1
                    loss = self.evaluate_batch(batch, stage=Stage.VALID)
                    out = self.compute_forward(batch, stage=stage)
                    loss = self.compute_objectives(out, batch, stage=stage)
                    loss.detach().cpu()
                    avg_valid_loss = self.update_average(loss, avg_valid_loss)
                self.step = 0
                self.on_stage_end(Stage.VALID, avg_valid_loss, epoch)

    def evaluate(self,test_set,max_key=None,min_key=None):
        test_set = self.make_dataloader()
        self.on_stage_start(Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        with torch.no_grad():
            for batch in test_set:
                self.step += 1
                out = self.compute_forward(batch, stage=stage)
                loss = self.compute_objectives(out, batch, stage=stage)
                loss.detach().cpu()
                avg_test_loss = self.update_average(loss, avg_test_loss)
            self.on_stage_end(Stage.TEST, avg_test_loss, None)
        self.step = 0
        return avg_test_loss

    
    