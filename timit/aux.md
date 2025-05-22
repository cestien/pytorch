<link href="/home/cestien/memoria/template/nota_template/miestilo.css" rel="stylesheet"></link>

# Notas antes de incluirlas en `notas.md`

## Main
``` python
    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, test_data = dataio_prep(hparams)

    # Trainer initialization
    asr_brain = ASR_Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    
    # Training/validation loop
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    asr_brain.evaluate(
        test_data,
        min_key="PER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

```
## Que cosas tiene cada método
```python
fit():
    make_dataloader()
    on_fit_start()
    _fit_train()

```

## `datio_prepare` simplificado
```python
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

```