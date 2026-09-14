# 현재 작업 디렉토리와 상관없이 공용 작업공간 경로를 찾는다.
from pathlib import Path as _WorkspacePath
import sys as _workspace_sys
_workspace_root = next(p for p in _WorkspacePath(__file__).resolve().parents if (p / 'workspace_paths.py').is_file())
_workspace_sys.path.insert(0, str(_workspace_root))
from workspace_paths import workspace_path

from pathlib import Path
import json, copy, datetime
import numpy as np
import cv2, torch, ultralytics, albumentations as A
from ultralytics.data.augment import Albumentations
from ultralytics.utils.instance import Instances
from ultralytics.cfg import get_cfg
from augmentation_runtime import training_transforms
R=workspace_path('KOREA_WASTE_RULES_20260913_v1')
def read(p): return json.loads((workspace_path('KOREA_WASTE_RULES_20260913_v1', p)).read_text('utf-8'))
def write(p,d): (workspace_path('KOREA_WASTE_RULES_20260913_v1', p)).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    transforms=training_transforms()
    get_cfg(overrides={'augmentations':transforms})
    wrapper=Albumentations(transforms=[A.to_dict(t) for t in transforms])
    assert wrapper.transform is not None and not wrapper.contains_spatial
    wrapper.transform.set_random_seed(73)
    p=Path((workspace_path('KOREA_WASTE_RULES_20260913_v1', 'finetune/bbox_train.txt')).read_text('utf-8').splitlines()[0])
    img=cv2.imread(str(p));assert img.shape==(640,640,3)
    sample={'img':img,'cls':np.array([[0]],dtype=np.float32),'instances':Instances(np.array([[.5,.5,.4,.4]],dtype=np.float32),np.array([[[.3,.3],[.7,.3],[.7,.7],[.3,.7]]],dtype=np.float32),bbox_format='xywh',normalized=True)}
    changed=0
    for _ in range(100):
        result=wrapper(copy.deepcopy(sample))
        assert result['img'].shape==img.shape
        assert np.array_equal(result['cls'],sample['cls'])
        assert np.array_equal(result['instances'].bboxes,sample['instances'].bboxes)
        assert np.array_equal(result['instances'].segments,sample['instances'].segments)
        changed+=not np.array_equal(result['img'],img)
    assert 0<changed<100
    assert torch.cuda.is_available()
    write('reports/runtime_augmentation_validation.json',{'passed':True,'trials':100,'changed_images':changed,'coordinates_and_classes_unchanged':True,'serialized_transforms_restored':True,'transforms':[A.to_dict(t) for t in transforms],'torch':torch.__version__,'ultralytics':ultralytics.__version__,'albumentations':A.__version__,'gpu':torch.cuda.get_device_name(0),'extra_image_files_created':0})
    final=read('reports/final_training_data_validation.json');mix=read('reports/finetune_mix.json')
    for task in ['bbox','seg']:
        for split in ['train','val','test']:
            paths=[Path(x) for x in (workspace_path('KOREA_WASTE_RULES_20260913_v1', f'finetune/{task}_{split}.txt')).read_text('utf-8').splitlines() if x]
            base=(workspace_path('KOREA_WASTE_RULES_20260913_v1', f'training_640_{task}/images/{split}')).resolve()
            assert all(p.resolve().parent==base and p.exists() for p in paths)
            final['counts'][f'{task}_{split}_selected_entries']=len(paths)
        assert final['counts'][f'{task}_train_selected_entries']==mix[task]['epoch_entries']
    final['sampling_rechecked_at']=datetime.datetime.now().isoformat()
    write('reports/final_training_data_validation.json',final)
    recovery=workspace_path('.', 'work/cleanup_recovery_20260913/staged_bbox')
    obsolete=[n for n in ['raw','prepared','staged_bbox','staged_seg','reviewed_bbox','review_pending_640','annotations'] if (workspace_path('KOREA_WASTE_RULES_20260913_v1', n)).exists()]
    assert not obsolete
    write('reports/cleanup_result.json',{'new_dataset_clean':True,'obsolete_directories_remaining':obsolete,'method':'Recycle Bin for removed sources, archives and previews; staged_bbox isolated outside new dataset after Windows recycling failed.','recovery_copy_remaining':str(recovery) if recovery.exists() else None,'permanent_delete_policy_rejected':True,'recycle_error':'Attempted to perform an unauthorized operation (staged_bbox)','original_project_datasets_kept':True,'training_images':final['images_checked'],'image_bytes':final['counts']['total_image_bytes'],'note':'Recycle Bin and recovery copy may still occupy disk space; not used for training.'})
    assert final['passed'] and final['all_image_metadata_removed']
    write('reports/training_ready.json',{'ready':True,'checked_at':datetime.datetime.now().isoformat(),'sequence':['bbox','seg'],'final_validation':'final_training_data_validation.json','runtime_validation':'runtime_augmentation_validation.json','note':'640 JPEG only; real general and packaging replay; native Mosaic plus pixel-only blur/compression.'})
    print(json.dumps({'ready':True,'changed_trials':changed,'bbox_entries':mix['bbox']['epoch_entries'],'seg_entries':mix['seg']['epoch_entries']}))
if __name__=='__main__':main()
