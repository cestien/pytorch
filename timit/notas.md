<link href="/home/cestien/memoria/template/nota_template/miestilo.css" rel="stylesheet"></link>

# Pseudo código y comentarios

Notas para poder implementar el código en pytorch.
```python
main()
    train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)
    ASR_Brain()
    fit()
    evaluate()

fit()
    train_set = self.make_dataloader(train_data)
    valid_set = self.make_dataloader(valid_data)
    for epoch in epoch.counter:
        # on stage_start
        ctc_metric = speechbrain.utils.metric_stats.MetricStats(metric=speechbrain.nnet.losses.ctc_loss)
        per_metric = speechbrain.utils.metric_stats.ErrorRateStats
```python
train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)
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

```