# Versión de train simplificada
class ASR_Brain(sb.Brain):
    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.ctc_loss, blank_index=<blank_index>, reduction:=batch).ctc_stats()
        if stage != sb.Stage.TRAIN:
            self.per_metrics = sb.utils.metric_stats.ErrorRateStats.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a stage."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")

        if stage == sb.Stage.VALID:
            lr_annealing = sb.nnet.schedulers.NewBobScheduler(initial_value=<lr>,improvement_threshold=0.0025, 
                                               annealing_factor=0.8, patient=0)
            old_lr, new_lr = lr_annealing(per)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            sb.utils.train_logger.FileTrainLogger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={"loss": stage_loss, "PER": per},
            )

        elif stage == sb.Stage.TEST:
            sb.utils.train_logger.FileTrainLogger.log_stats(
                stats_meta={"Epoch loaded": sb.utils.epoch_loop.EpochCounter.current},
                test_stats={"loss": stage_loss, "PER": per},
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w", encoding="utf-8") as w:
                    w.write("CTC loss stats:\n")
                    self.ctc_metrics.write_stats(w)
                    w.write("\nPER stats:\n")
                    self.per_metrics.write_stats(w)
                    print("CTC and PER stats written to ",self.hparams.test_wer_file)

    def compute_forward(self, batch, stage):
            "Given an input batch it computes the phoneme probabilities."
            batch = batch.to(self.device)
            wavs, wav_lens = batch.sig

            # Add waveform augmentation if specified.
            wav_augment = sb.augment.augmenter.Augmenter(concat_original=True,min_augmentations=4,max_augmentations=4,
                                        augment_prob=1.0,augmentations=[add_noise,speed_perturb,drop_freq,drop_chunk])
        
            if stage == sb.Stage.TRAIN and hasattr(wav_augment):
                wavs, wav_lens = wav_augment(wavs, wav_lens)
            compute_features = sb.lobes.features.Fbank()
            log_softmax = sb.nnet.activations.Softmax()
            feats = compute_features(wavs)
            feats = self.modules.normalize(feats, wav_lens)
            out = self.modules.model(feats)
            out = self.modules.output(out)
            pout = self.hparams.log_softmax(out)

            return pout, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the CTC loss."
        pout, pout_lens = predictions
        phns, phn_lens = batch.phn_encoded

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            phns = self.hparams.wav_augment.replicate_labels(phns)
            phn_lens = self.hparams.wav_augment.replicate_labels(phn_lens)
        compute_cost = sb.nnet.losses.ctc_loss()
        loss = compute_cost(pout, phns, pout_lens, phn_lens)
        self.ctc_metrics.append(batch.id, pout, phns, pout_lens, phn_lens)

        if stage != sb.Stage.TRAIN:
            sequence = sb.decoders.ctc_greedy_decode(
                pout, pout_lens, blank_id=self.hparams.blank_index
            )
            self.per_metrics.append(
                ids=batch.id,
                predict=sequence,
                target=phns,
                target_len=phn_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )

        return loss

def dataio_prep(hparams):
    # Crea los datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path)
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path)
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path)
    # Los ordenamos por duración (default sin ordernar)
    train_data = train_data.filtered_sorted(sort_key="duration")
    valid_data = valid_data.filtered_sorted(sort_key="duration")
    test_data = test_data.filtered_sorted(sort_key="duration")
    datasets = [train_data, valid_data, test_data]
    # Pipeline de los wav
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    # Pipeline de los labels
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded = sb.dataio.encoder.CTCTextEncoder().encode_sequence_torch(phn_list)
        yield phn_encoded
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    return train_data, valid_data, test_data

if __name__ == "__main__":
    train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)
    asr_brain = ASR_Brain()
    asr_brain.label_encoder = label_encoder
    asr_brain.fit(epoch_counter,train_data,valid_data)
    asr_brain.evaluate(test_data,min_key="PER")