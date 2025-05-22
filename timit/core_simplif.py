

class Brain:
    """Brain class abstracts away the details of data loops.

    The primary purpose of the `Brain` class is the implementation of
    the ``fit()`` method, which iterates epochs and datasets for the
    purpose of "fitting" a set of modules to a set of data.

    In order to use the ``fit()`` method, one should sub-class the ``Brain``
    class and override any methods for which the default behavior does not
    match the use case. For a simple use case (e.g., training a single model
    with a single dataset) the only methods that need to be overridden are:

    * ``compute_forward()``
    * ``compute_objectives()``

    The example below illustrates how overriding these two methods is done.

    For more complicated use cases, such as multiple modules that need to
    be updated, the following methods can be overridden:

    * ``fit_batch()``
    * ``evaluate_batch()``

    Arguments
    ---------
    modules : dict of str:torch.nn.Module pairs
        These modules are passed to the optimizer by default if they have
        trainable parameters, and will have ``train()``/``eval()`` called on them.
    opt_class : torch.optim class
        A torch optimizer constructor that takes only the list of
        parameters (e.g. a lambda or partial function definition). By default,
        this will be passed all modules in ``modules`` at the
        beginning of the ``fit()`` method. This behavior can be changed
        by overriding the ``configure_optimizers()`` method.
    hparams : dict
        Each key:value pair should consist of a string key and a hyperparameter
        that is used within the overridden methods. These will
        be accessible via an ``hparams`` attribute, using "dot" notation:
        e.g., self.hparams.model(x).
    run_opts : dict
        A set of options to change the runtime environment, including

        debug (bool)
            If ``True``, this will only iterate a few batches for all
            datasets, to ensure code runs without crashing.
        debug_batches (int)
            Number of batches to run in debug mode, Default ``2``.
        debug_epochs (int)
            Number of epochs to run in debug mode, Default ``2``.
            If a non-positive number is passed, all epochs are run.
        debug_persistently (bool)
            Keep data stored during debug mode (not using /tmp), Default ``False``.
        jit (bool)
            Enable to compile all modules using jit, Default ``False``.
        jit_module_keys (list of str)
            List of keys in ``modules`` that should be jit compiled.
        compile (bool)
            Enable to compile all modules using torch.compile, Default ``False``.
        compile_module_keys (list of str)
            List of keys in ``modules`` that should be compiled using
            ``torch.compile``. If ``torch.compile`` is unavailable,
            an error is raised.
        compile_mode (str)
            One of ``default``, ``reduce-overhead``, ``max-autotune``, Default ``reduce-overhead``.
        compile_using_fullgraph (bool)
            Whether it is ok to break model into several subgraphs, Default ``False``.
        compile_using_dynamic_shape_tracing (bool)
            Use dynamic shape tracing for compilation, Default ``False``.
        distributed_backend (str)
            One of ``nccl``, ``gloo``, ``mpi``.
        device (str)
            The location for performing computations.
        precision (str)
            One of ``fp32``, ``fp16``, ``bf16``.
        eval_precision (str)
            One of ``fp32``, ``fp16``, ``bf16``.
        auto_mix_prec (bool)
            If ``True``, automatic mixed-precision (fp16) is used.
            Activate it only with cuda. Note: this is a
            deprecated feature, and will be removed in the future.
        bfloat16_mix_prec (bool)
            If ``True``, automatic mixed-precision (bf16) is used.
            Activate it only with cuda. Note: this is a
            deprecated feature, and will be removed in the future.
        max_grad_norm (float)
            Default implementation of ``fit_batch()`` uses
            ``clip_grad_norm_`` with this value. Default: ``5``.
        skip_nonfinite_grads (bool)
            If ``True``, sets gradients to zero if they are non-finite
            (e.g., NaN, Inf). Default: ``False``.
        nonfinite_patience (int)
            Number of times to ignore non-finite losses before stopping.
            Default: ``3``.
        noprogressbar (bool)
            Whether to turn off progressbar when training. Default: ``False``.
        ckpt_interval_minutes (float)
            Amount of time between saving intra-epoch checkpoints,
            in minutes, default: ``15.0``. If non-positive, these are not saved.
        ckpt_interval_steps (int)
            Number of steps between saving intra-epoch checkpoints.
            If non-positive, these are not saved. Default: ``0``.


        Typically in a script this comes from ``speechbrain.parse_args``, which
        has different defaults than Brain. If an option is not defined here
        (keep in mind that parse_args will inject some options by default),
        then the option is also searched for in hparams (by key).
    checkpointer : speechbrain.Checkpointer
        By default, this will be used to load checkpoints, and will have the
        optimizer added to continue training if interrupted.

    Example
    -------
    >>> from torch.optim import SGD
    >>> class SimpleBrain(Brain):
    ...     def compute_forward(self, batch, stage):
    ...         return self.modules.model(batch[0])
    ...     def compute_objectives(self, predictions, batch, stage):
    ...         return torch.nn.functional.l1_loss(predictions, batch[0])
    >>> model = torch.nn.Linear(in_features=10, out_features=10)
    >>> brain = SimpleBrain({"model": model}, opt_class=lambda x: SGD(x, 0.1))
    >>> brain.fit(range(1), ([torch.rand(10, 10), torch.rand(10, 10)],))
    """

    def __init__(self,modules=None,opt_class=None,self.optimizers_dict = None):

        for arg, default in run_opt_defaults.items():
            if run_opts is not None and arg in run_opts:
                if hparams is not None and arg in hparams:
                    logger.info(
                        "Info: "
                        + arg
                        + " arg overridden by command line input to: "
                        + str(run_opts[arg])
                    )
                setattr(self, arg, run_opts[arg])
            else:
                # If any arg from run_opt_defaults exist in hparams and
                # not in command line args "run_opts"
                if hparams is not None and arg in hparams:
                    logger.info(
                        "Info: " + arg + " arg from hparam file is used"
                    )
                    setattr(self, arg, hparams[arg])
                else:
                    setattr(self, arg, default)

        # Check Python version
        if not (
            sys.version_info.major == PYTHON_VERSION_MAJOR
            and sys.version_info.minor >= PYTHON_VERSION_MINOR
        ):
            logger.warning(
                "Detected Python "
                + str(sys.version_info.major)
                + "."
                + str(sys.version_info.minor)
                + ". We suggest using SpeechBrain with Python >="
                + str(PYTHON_VERSION_MAJOR)
                + "."
                + str(PYTHON_VERSION_MINOR)
            )

        # Assume `torchrun` was used if `RANK` and `LOCAL_RANK` are set
        self.distributed_launch = (
            os.environ.get("RANK") is not None
            and os.environ.get("LOCAL_RANK") is not None
        )

        if self.data_parallel_backend and self.distributed_launch:
            raise ValueError(
                "To use data_parallel backend, start your script with:\n\t"
                "python experiment.py hyperparams.yaml "
                "--data_parallel_backend=True\n"
                "To use DDP backend, start your script with:\n\t"
                "torchrun [args] experiment.py hyperparams.yaml"
            )

        if self.ckpt_interval_minutes > 0 and self.ckpt_interval_steps > 0:
            sys.exit(
                "The options `ckpt_interval_minutes` and `ckpt_interval_steps` "
                "are mutually exclusive. "
                "Please keep only one active per experiment run."
            )

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

        # Put modules on the right device, accessible with dot notation
        self.modules = torch.nn.ModuleDict(modules).to(self.device)

        # The next line ensures that both tensors marked as parameters and standard tensors,
        # such as those used in InputNormalization, are placed on the right device.
        for module in self.modules:
            if hasattr(self.modules[module], "to"):
                self.modules[module] = self.modules[module].to(self.device)

        # Make hyperparams available with dot notation too
        if hparams is not None:
            self.hparams = SimpleNamespace(**hparams)

        # Checkpointer should point at a temporary directory in debug mode
        if (
            self.debug
            and not self.debug_persistently
            and self.checkpointer is not None
            and hasattr(self.checkpointer, "checkpoints_dir")
        ):
            tempdir = tempfile.TemporaryDirectory()
            logger.info(
                "Since debug mode is active, switching checkpointer "
                f"output to temporary directory: {tempdir.name}"
            )
            self.checkpointer.checkpoints_dir = pathlib.Path(tempdir.name)

            # Keep reference to tempdir as long as checkpointer exists
            self.checkpointer.tempdir = tempdir

        # Sampler should be handled by `make_dataloader`
        # or if you provide a DataLoader directly, you can set
        # this.train_sampler = your_sampler
        # to have your_sampler.set_epoch() called on each epoch.
        self.train_sampler = None

        if self.auto_mix_prec:
            logger.warning(
                "The option `--auto_mix_prec` is deprecated and will be removed in the future. "
                "Please use `--precision=fp16` instead."
            )
            self.precision = "fp16"

        if self.bfloat16_mix_prec:
            logger.warning(
                "The option `--bfloat16_mix_prec` is deprecated and will be removed in the future. "
                "Please use `--precision=bf16` instead."
            )
            self.precision = "bf16"

        if self.device_type == "cpu" and (
            self.precision == "fp16" or self.eval_precision == "fp16"
        ):
            raise ValueError(
                "The option `--precision` or `--eval_precision` is set to fp16. "
                "This option is not yet supported on CPU. "
                "Please use `--precision=bf16` or `--eval_precision=bf16` instead "
                "to enable mixed precision on CPU."
            )

        gradscaler_enabled = (
            self.precision == "fp16" and self.device_type == "cuda"
        )
        if self.skip_nonfinite_grads and gradscaler_enabled:
            logger.warning(
                "The option `skip_nonfinite_grads` will be ignored "
                "because GradScaler is enabled and will automatically "
                "skip nonfinite gradients."
            )

        logger.info(f"Gradscaler enabled: `{gradscaler_enabled}`")
        logger.info(f"Using training precision: `--precision={self.precision}`")
        logger.info(
            f"Using evaluation precision: `--eval_precision={self.eval_precision}`"
        )
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.scaler = torch.cuda.amp.GradScaler(enabled=gradscaler_enabled)
        else:
            self.scaler = torch.GradScaler(
                self.device, enabled=gradscaler_enabled
            )

        train_dtype = AMPConfig.from_name(self.precision).dtype
        self.training_ctx = TorchAutocast(
            device_type=self.device_type, dtype=train_dtype
        )
        eval_dtype = AMPConfig.from_name(self.eval_precision).dtype
        self.evaluation_ctx = TorchAutocast(
            device_type=self.device_type, dtype=eval_dtype
        )
        if gradscaler_enabled and self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "scaler", self.scaler, optional_load=True
            )

        # List parameter count for the user
        self.print_trainable_parameters()

        if self.distributed_launch:
            self.rank = int(os.environ["RANK"])
            if not is_distributed_initialized():
                if self.rank > 0:
                    raise ValueError(
                        " ================ WARNING ==============="
                        "Please add sb.ddp_init_group() into your exp.py"
                        "To use DDP backend, start your script with:\n\t"
                        "torchrun [args] experiment.py hyperparams.yaml"
                    )
                else:
                    logger.warning(
                        "To use DDP, please add "
                        "sb.utils.distributed.ddp_init_group() into your exp.py"
                    )
                    logger.info(
                        "Only the main process is alive, "
                        "all other subprocess were killed."
                    )

        # Prepare iterating variables
        self.avg_train_loss = 0.0
        self.step = 0
        self.optimizer_step = 0

        # Add this class to the checkpointer for intra-epoch checkpoints
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("brain", self)

        # Force default color for tqdm progressbar
        if not self.tqdm_colored_bar:
            self.tqdm_barcolor = dict.fromkeys(self.tqdm_barcolor, "")

        # Profiler setup
        self.profiler = None
        if self.profile_training:
            logger.info("Pytorch profiler has been activated.")
            self.tot_prof_steps = (self.profile_steps + self.profile_warmup) - 1
            self.profiler = prepare_profiler(
                self.profile_warmup,
                self.profile_steps,
                self.hparams.output_folder,
            )

    def compute_forward(self, batch, stage):
        """Forward pass, to be overridden by sub-classes.

        Arguments
        ---------
        batch : torch.Tensor or tensors
            An element from the dataloader, including inputs for processing.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST

        Returns
        -------
        torch.Tensor or torch.Tensors
            The outputs after all processing is complete.
            Directly passed to ``compute_objectives()``.
        """
        raise NotImplementedError
        return

    def compute_objectives(self, predictions, batch, stage):
        """Compute loss, to be overridden by sub-classes.

        Arguments
        ---------
        predictions : torch.Tensor or torch.Tensors
            The output tensor or tensors to evaluate.
            Comes directly from ``compute_forward()``.
        batch : torch.Tensor or tensors
            An element from the dataloader, including targets for comparison.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST

        Returns
        -------
        loss : torch.Tensor
            A tensor with the computed loss.
        """
        raise NotImplementedError
        return

    def on_stage_start(self, stage, epoch=None):
        """Gets called when a stage starts.

        Useful for defining class variables used during the stage.

        Arguments
        ---------
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST
        epoch : int
            The current epoch count.
        """
        pass

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage.

        Useful for computing stage statistics, saving checkpoints, etc.

        Arguments
        ---------
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST
        stage_loss : float
            The average loss over the completed stage.
        epoch : int
            The current epoch count.
        """
        pass

    def make_dataloader(self, dataset, stage, ckpt_prefix="dataloader-", **loader_kwargs):
        """Creates DataLoaders for Datasets.

        This is used by ``fit()`` and ``evaluate()`` if they just receive
        Datasets.

        Alternatively, this can be called from outside the Brain subclass.
        In that case, the DataLoader should be passed to ``fit()`` in place
        of the dataset.

        The Stage.TRAIN DataLoader is handled specially. It has extra args for
        shuffle and drop_last. In DDP a DistributedSampler is created (unless
        the dataset is an IterableDataset).

        NOTE
        ----
        Some important DataLoader arguments are passed via **loader_kwargs,
        e.g., batch_size, num_workers, pin_memory.

        NOTE
        ----
        By default, ``evaluate()`` specifies ckpt_prefix=None to stop the test
        DataLoader being added to the checkpointer. If you need to add a
        recoverable after saving checkpoints (e.g., at test time, after
        checkpointing the training), and still be able to recover reasonably,
        you should probably specify ``allow_partial_load=True``.

        Arguments
        ---------
        dataset : Dataset
            A set of data to use to create data loader. If the Dataset is a
            DynamicItemDataset, PaddedBatch is used as the default collate_fn,
            unless specified in loader_kwargs.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST
        ckpt_prefix : str, None
            Prefix to use for SaveableDataLoader Checkpoint name. The Stage
            name is added to this to create the full key. Set to None to not
            save the DataLoader.
        **loader_kwargs : dict
            Additional keyword arguments to the DataLoader.
            E.g., batch_size, num_workers, pin_memory.

        Returns
        -------
        DataLoader for the input dataset
        """
        dataloader = sb.dataio.dataloader.make_dataloader(dataset, **loader_kwargs)
        return dataloader

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """

        # Initialize optimizers 
        opt_class = torch.optim.Adadelta(rho=0.95,lr = 'lr',eps=1.e-8)

    
    def zero_grad(self, set_to_none=False):
        """Sets the gradients of all optimized ``torch.Tensor``s to zero
        if ``set_to_none=False`` (default) or to None otherwise.

        Setting gradients to None should save the memory, e.g.
        during ``evaluate()`` and thus larger batch might be used.
        """
        if self.optimizers_dict is not None:
            for opt in self.freeze_optimizers(self.optimizers_dict).values():
                opt.zero_grad(set_to_none=set_to_none)
        elif self.opt_class is not None:
            self.optimizer.zero_grad(set_to_none=set_to_none)

    def on_evaluate_start(self, max_key=None, min_key=None):
        """Gets called at the beginning of ``evaluate()``

        Default implementation loads the best-performing checkpoint for
        evaluation, based on stored metrics.

        Arguments
        ---------
        max_key : str
            Key to use for finding best checkpoint (higher is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        min_key : str
            Key to use for finding best checkpoint (lower is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        """

        # Recover best checkpoint for evaluation
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                max_key=max_key, min_key=min_key
            )

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``
        * ``optimizers_step()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        should_step = (self.step % self.grad_accumulation_factor) == 0
        self.on_fit_batch_start(batch, should_step)

        with self.no_sync(not should_step):
            with self.training_ctx:
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            scaled_loss = self.scaler.scale(
                loss / self.grad_accumulation_factor
            )
            self.check_loss_isfinite(scaled_loss)
            scaled_loss.backward()

        if should_step:
            self.optimizers_step()

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()

    def optimizers_step(self):
        """Performs a step of gradient descent on the optimizers. This method is called every
        ``grad_accumulation_factor`` steps."""
        # 1. get the valid optimizers, i.e., the ones that are not frozen during this step
        if self.optimizers_dict is not None:
            valid_optimizers = self.freeze_optimizers(self.optimizers_dict)
        elif self.opt_class is not None:
            # if valid_optimizers is not defined which could happen if a user is using an old
            # init_optimizers() method, then we assume that the only valid optimizer is
            # self.optimizer (which is the default behavior).
            valid_optimizers = {"optimizer": self.optimizer}
        else:
            # Note: in some cases you might want to only compute gradients statistics and
            # you do not need to call the optimizers.step() method. In this case, you can
            # simply return from this method and skip the rest of the code.
            return

        # 2. unscale the gradients of the valid optimizers
        for opt in valid_optimizers.values():
            self.scaler.unscale_(opt)

        # 3. clip gradients
        # We are clipping this way because clipping on self.modules.parameters()
        # can leads to NaN/Inf gradients norm as doing the concatenation
        # of all parameters in a single vector can lead to overflow/underflow.
        for opt in valid_optimizers.values():
            torch.nn.utils.clip_grad_norm_(
                opt.param_groups[0]["params"], self.max_grad_norm
            )

        # Note: no need to activate this flag if you are in fp16
        # since GradScaler is automatically handling the nonfinite gradients
        if not self.scaler.is_enabled() and self.skip_nonfinite_grads:
            self.check_gradients()

        # 4. step the valid optimizers
        # If the scaler is disable, it simply calls optimizer.step()
        for opt in valid_optimizers.values():
            self.scaler.step(opt)

        self.scaler.update()

        for opt in valid_optimizers.values():
            opt.zero_grad(set_to_none=True)

        self.optimizer_step += 1

    def on_fit_batch_start(self, batch, should_step):
        """Called at the beginning of ``fit_batch()``.

        This method is not called under the AMP context manager. Do not assume
        automatic casting of the input batch to a lower precision (e.g. fp16).

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        should_step : boolean
            Whether optimizer.step() was called or not.
        """
        pass

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Called after ``fit_batch()``.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        outputs : list or dictionary of torch.Tensors
            Returned value of compute_forward().
        loss : torch.Tensor
            Returned value of compute_objectives().
        should_step : boolean
            Whether optimizer.step() was called or not.
        """
        pass

    def evaluate_batch(self, batch, stage):
        """Evaluate one batch, override for different procedure than train.

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """
        with self.evaluation_ctx:
            out = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(out, batch, stage=stage)
        return loss.detach().cpu()

    def _fit_train(self, train_set, epoch, enable):
        # Training stage
        self.on_stage_start(Stage.TRAIN, epoch)
        self.modules.train()
        self.zero_grad()

        # Reset nonfinite count to 0 each epoch
        self.nonfinite_count = 0

        if self.train_sampler is not None and hasattr(
            self.train_sampler, "set_epoch"
        ):
            self.train_sampler.set_epoch(epoch)

        # Time since last intra-epoch checkpoint
        last_ckpt_time = time.time()
        steps_since_ckpt = 0
        with tqdm(
            train_set,
            initial=self.step,
            dynamic_ncols=True,
            disable=not enable,
            colour=self.tqdm_barcolor["train"],
        ) as t:
            if self.profiler is not None:
                self.profiler.start()
            for batch in t:
                if self._optimizer_step_limit_exceeded:
                    logger.info("Train iteration limit exceeded")
                    break
                self.step += 1
                steps_since_ckpt += 1
                loss = self.fit_batch(batch)
                self.avg_train_loss = self.update_average(
                    loss, self.avg_train_loss
                )
                t.set_postfix(train_loss=self.avg_train_loss)

                if self.profiler is not None:
                    self.profiler.step()
                    if self.profiler.step_num > self.tot_prof_steps:
                        logger.info(
                            "The profiler finished, training is stopped."
                        )
                        self.profiler.stop()
                        quit()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

                if self._should_save_intra_epoch_ckpt(
                    last_ckpt_time, steps_since_ckpt
                ):
                    # Checkpointer class will handle running this on main only
                    self._save_intra_epoch_ckpt()
                    last_ckpt_time = time.time()
                    steps_since_ckpt = 0

        # Run train "on_stage_end" on all processes
        self.zero_grad(set_to_none=True)  # flush gradients
        self.on_stage_end(Stage.TRAIN, self.avg_train_loss, epoch)
        self.avg_train_loss = 0.0
        self.step = 0

    def _fit_valid(self, valid_set, epoch, enable):
        # Validation stage
        if valid_set is not None:
            self.on_stage_start(Stage.VALID, epoch)
            self.modules.eval()
            avg_valid_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(
                    valid_set,
                    dynamic_ncols=True,
                    disable=not enable,
                    colour=self.tqdm_barcolor["valid"],
                ):
                    self.step += 1
                    loss = self.evaluate_batch(batch, stage=Stage.VALID)
                    avg_valid_loss = self.update_average(loss, avg_valid_loss)

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                self.step = 0
                self.on_stage_end(Stage.VALID, avg_valid_loss, epoch)

    def fit(self,epoch_counter,train_set,valid_set=None,):
        if sb.Stage.TRAIN:
            train_set = self.make_dataloader(train_set, stage=sb.Stage.TRAIN)
        if sb.Stage.VALID:
            valid_set = self.make_dataloader(valid_set,stage=sb.Stage.VALID)
                
        self.on_fit_start()

        # Iterate epochs
        for epoch in epoch_counter:
            self._fit_train(train_set=train_set, epoch=epoch)
            self._fit_valid(valid_set=valid_set, epoch=epoch)

    def evaluate(self,test_set,max_key=None,min_key=None):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                test_set,
                dynamic_ncols=True,
                disable=not enable,
                colour=self.tqdm_barcolor["test"],
            ):
                self.step += 1
                loss = self.evaluate_batch(batch, stage=Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            self.on_stage_end(Stage.TEST, avg_test_loss, None)
        self.step = 0
        return avg_test_loss

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

    