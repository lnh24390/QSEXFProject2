from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'workspace_paths.py').exists())
sys.path[:0] = [str(ROOT), str(ROOT/'02_training')]
from model_analysis.schema import summary, compare_records, write_json
from model_analysis.evaluation import class_mapping, remapped_validator, dataset_snapshot, validate_task_labels
from model_analysis.history import training_summary


class AnalysisTests(unittest.TestCase):
    def test_external_class_order_and_explicit_aliases(self):
        self.assertEqual(class_mapping({0:'glass',1:'paper'}, {0:'paper',1:'glass'}), {0:1,1:0})
        self.assertEqual(class_mapping({0:'class0',1:'class1'}, {0:'paper',1:'glass'},
                                      {'0':'glass','1':'paper'}), {0:1,1:0})
        with self.assertRaises(ValueError):
            class_mapping({0:'person',1:'car'}, {0:'paper',1:'glass'})
        with self.assertRaises(ValueError):
            class_mapping({0:'a',1:'b'}, {0:'paper',1:'glass'}, {'a':'paper','b':'paper'})

    def test_mapping_changes_predictions_not_boxes_or_masks(self):
        import torch
        from ultralytics.models.yolo.detect import DetectionValidator
        from ultralytics.models.yolo.segment import SegmentationValidator
        for task, base in [('detect', DetectionValidator), ('segment', SegmentationValidator)]:
            boxes = torch.ones(2, 4); masks = torch.ones(2, 8, 8)
            predictions = [{'cls':torch.tensor([0.,1.]), 'bboxes':boxes, 'masks':masks}]
            klass = remapped_validator(task, {0:1,1:0}, {0:'paper',1:'glass'})
            with patch.object(base, 'postprocess', return_value=predictions):
                result = klass.__new__(klass).postprocess(None)
            self.assertEqual(result[0]['cls'].tolist(), [1.,0.])
            self.assertIs(result[0]['bboxes'], boxes)
            self.assertIs(result[0]['masks'], masks)

    def test_absent_class_does_not_inherit_overall_ap(self):
        metric = SimpleNamespace(ap_class_index=[1],p=[.8],r=[.7],ap50=[.6],ap=[.5],map=.5,map50=.6,mp=.8,mr=.7)
        result = SimpleNamespace(names={0:'paper',1:'glass'}, speed={'inference':1.2},results_dict={},box=metric)
        record = summary(result)
        self.assertIsNone(record['box']['per_class']['paper']['map'])
        self.assertEqual(record['box']['per_class_map'], [None,.5])

    def test_comparison_rejects_different_dataset_or_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_json(folder/'a.json', {'comparison_key':'dataset_A'})
            write_json(folder/'b.json', {'comparison_key':'dataset_B'})
            with self.assertRaises(ValueError):
                compare_records([folder/'a.json',folder/'b.json'], folder/'outputs')
            self.assertFalse((folder/'outputs').exists())

    def test_dataset_fingerprint_detects_same_size_label_change(self):
        import yaml
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            images = folder/'images'/'test'; labels = folder/'labels'/'test'
            images.mkdir(parents=True);labels.mkdir(parents=True)
            for i in range(2):
                Image.new('RGB',(32,32),(i*100,0,0)).save(images/f'{i}.png')
                (labels/f'{i}.txt').write_text('0 0.5 0.5 0.2 0.2\n')
            config = folder/'data.yaml'
            config.write_text(yaml.safe_dump(dict(path=str(folder),train=str(images),val=str(images),test=str(images),names={0:'paper',1:'glass'})))
            before, _ = dataset_snapshot(config, 'test')
            self.assertEqual(before['count'],2)
            (labels/'0.txt').write_text('1 0.5 0.5 0.2 0.2\n')
            after, _ = dataset_snapshot(config, 'test')
            self.assertNotEqual(before['fingerprint'], after['fingerprint'])

    def test_training_summary_uses_mask_fitness_and_real_csv_epochs(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'results.csv').write_text('epoch,time,metrics/mAP50-95(B),metrics/mAP50-95(M),train/seg_loss\n1,100,0.9,0.1,1.5\n2,200,0.8,0.5,1.2\n3,50,0.7,0.4,1.1\n')
            result = training_summary(folder)
            self.assertEqual(result['recorded_epochs'],3)
            self.assertEqual(result['best_epoch'],2)
            self.assertEqual(result['timer_resets'],1)
            self.assertEqual(result['mean_epoch_seconds_excluding_start_or_reset'],100)
            self.assertEqual(result['last_losses']['train/seg_loss'],1.1)

    def test_polygon_and_box_annotations_are_not_interchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            label = Path(temp)/'label.txt'
            label.write_text('0 0.5 0.5 0.2 0.2\n')
            validate_task_labels([{'label':str(label)}], 'detect')
            with self.assertRaises(ValueError):
                validate_task_labels([{'label':str(label)}], 'segment')

    def test_baseline_logger_uses_real_loss_names(self):
        import importlib.util
        import torch
        spec = importlib.util.spec_from_file_location('reviewed_logger', ROOT/'02_training/baseline_training/metrics_logger.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        callbacks = {}
        model = SimpleNamespace(add_callback=lambda name, func: callbacks.setdefault(name, []).append(func))
        module.attach(model)
        with tempfile.TemporaryDirectory() as temp, patch.object(torch.cuda, 'is_available', return_value=False):
            folder = Path(temp)
            args = SimpleNamespace(model='test.pt', data='data.yaml', imgsz=640, batch=1, workers=0,
                cache=False,amp=False,epochs=100,patience=10,optimizer='AdamW',seed=0,device='cpu')
            trainer = SimpleNamespace(save_dir=folder,start_epoch=0,args=args,data={'names':{0:'paper'}},
                loss_names=('box_loss','seg_loss','cls_loss','dfl_loss'),epoch=0,tloss=None,
                metrics={'metrics/mAP50-95(B)':.4,'metrics/mAP50-95(M)':.3},
                label_loss_items=lambda _:dict(zip(('train/box_loss','train/seg_loss','train/cls_loss','train/dfl_loss'),(1,2,3,4))),
                optimizer=SimpleNamespace(param_groups=[{'lr':.001}]))
            callbacks['on_train_start'][0](trainer)
            (folder/'results.csv').write_text('epoch,time\n1,10\n')
            callbacks['on_fit_epoch_end'][0](trainer)
            text=(folder/'metrics.txt').read_text('utf-8')
            self.assertEqual(text.count('seg_loss'),1)
            self.assertIn('2.0000',text)
            trainer.epoch=1
            callbacks['on_fit_epoch_end'][0](trainer)
            self.assertEqual(text,(folder/'metrics.txt').read_text('utf-8'))


if __name__ == '__main__':
    unittest.main()
