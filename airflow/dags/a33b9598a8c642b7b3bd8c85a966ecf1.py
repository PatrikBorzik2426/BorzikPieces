from datetime import datetime
from dateutil.parser import parse
from airflow import DAG
from domino.task import Task

dag_config_0 = {'start_date': '2025-12-09T09:57:00', 'schedule': '@once', 'catchup': False, 'dag_id': 'a33b9598a8c642b7b3bd8c85a966ecf1'}

# Parse datetime values
dt_keys = ['start_date', 'end_date']
dag_config = { k: (v if k not in dt_keys else parse(v)) for k, v in dag_config_0.items()}
dag_config = {**dag_config, 'is_paused_upon_creation': False}

with DAG(**dag_config) as dag:
    NiftiDataL_37b84931888e454aac0ad9800b9a0106 = Task(
        dag,
        task_id='NiftiDataL_37b84931888e454aac0ad9800b9a0106',
        workspace_id=2,
        workflow_shared_storage={'source': 'Local', 'mode': 'Read/Write', 'provider_options': {}},
        container_resources={'requests': {'cpu': '100.0m', 'memory': '128.0Mi'}, 'limits': {'cpu': '100.0m', 'memory': '128.0Mi'}, 'use_gpu': False},
        piece={'name': 'NiftiDataLoaderPiece', 'source_image': 'ghcr.io/borzikpieces/borzikpieces:0.1.44-group0', 'repository_url': 'https://github.com/PatrikBorzik2426/BorzikPieces', 'repository_version': '0.1.44'},
        piece_input_kwargs={'images_path': '/home/shared_storage/medical_data/images', 'masks_path': '/home/shared_storage/medical_data/masks', 'file_pattern': '*.nii.gz'}
    )()
    NiftiVisua_f6bab4518ef3452eb58ef350fbda4ed9 = Task(
        dag,
        task_id='NiftiVisua_f6bab4518ef3452eb58ef350fbda4ed9',
        workspace_id=2,
        workflow_shared_storage={'source': 'Local', 'mode': 'Read/Write', 'provider_options': {}},
        container_resources={'requests': {'cpu': '100.0m', 'memory': '128.0Mi'}, 'limits': {'cpu': '100.0m', 'memory': '128.0Mi'}, 'use_gpu': False},
        piece={'name': 'NiftiVisualizationPiece', 'source_image': 'ghcr.io/borzikpieces/borzikpieces:0.1.44-group0', 'repository_url': 'https://github.com/PatrikBorzik2426/BorzikPieces', 'repository_version': '0.1.44'},
        piece_input_kwargs={'images_path': '/home/shared_storage/medical_data/images', 'masks_path': '/home/shared_storage/medical_data/masks', 'file_pattern': '*.nii.gz', 'max_subjects': 50, 'view_plane': 'axial', 'show_mask_overlay': True, 'mask_alpha': 0.5, 'color_map': 'gray', 'grid_columns': 3}
    )()
    DataSplitP_54acc3ceb29a423892a76ed1ae55c73d = Task(
        dag,
        task_id='DataSplitP_54acc3ceb29a423892a76ed1ae55c73d',
        workspace_id=2,
        workflow_shared_storage={'source': 'Local', 'mode': 'Read/Write', 'provider_options': {}},
        container_resources={'requests': {'cpu': '100.0m', 'memory': '128.0Mi'}, 'limits': {'cpu': '100.0m', 'memory': '128.0Mi'}, 'use_gpu': False},
        piece={'name': 'DataSplitPiece', 'source_image': 'ghcr.io/borzikpieces/borzikpieces:0.1.44-group0', 'repository_url': 'https://github.com/PatrikBorzik2426/BorzikPieces', 'repository_version': '0.1.44'},
        piece_input_kwargs={'subjects': {'type': 'fromUpstream', 'upstream_task_id': 'NiftiDataL_37b84931888e454aac0ad9800b9a0106', 'output_arg': 'subjects'}, 'train_ratio': 0.7, 'val_ratio': 0.15, 'test_ratio': 0.15, 'random_seed': 42, 'split_strategy': 'random'}
    )()
    NiftiPrepr_5a6c68d92d3642c7823630abcd6f20e6 = Task(
        dag,
        task_id='NiftiPrepr_5a6c68d92d3642c7823630abcd6f20e6',
        workspace_id=2,
        workflow_shared_storage={'source': 'Local', 'mode': 'Read/Write', 'provider_options': {}},
        container_resources={'requests': {'cpu': '100.0m', 'memory': '128.0Mi'}, 'limits': {'cpu': '100.0m', 'memory': '128.0Mi'}, 'use_gpu': False},
        piece={'name': 'NiftiPreprocessingPiece', 'source_image': 'ghcr.io/borzikpieces/borzikpieces:0.1.44-group0', 'repository_url': 'https://github.com/PatrikBorzik2426/BorzikPieces', 'repository_version': '0.1.44'},
        piece_input_kwargs={'subjects': {'type': 'fromUpstream', 'upstream_task_id': 'DataSplitP_54acc3ceb29a423892a76ed1ae55c73d', 'output_arg': 'val_subjects'}, 'output_dir': '/home/shared_storage/medical_data/preprocessed', 'normalization': 'zscore', 'lower_percentile': 1, 'upper_percentile': 99, 'save_as_numpy': True}
    )()
    NiftiPrepr_0e19aa0d1c584cfab52a7428f142b25e = Task(
        dag,
        task_id='NiftiPrepr_0e19aa0d1c584cfab52a7428f142b25e',
        workspace_id=2,
        workflow_shared_storage={'source': 'Local', 'mode': 'Read/Write', 'provider_options': {}},
        container_resources={'requests': {'cpu': '100.0m', 'memory': '128.0Mi'}, 'limits': {'cpu': '100.0m', 'memory': '128.0Mi'}, 'use_gpu': False},
        piece={'name': 'NiftiPreprocessingPiece', 'source_image': 'ghcr.io/borzikpieces/borzikpieces:0.1.44-group0', 'repository_url': 'https://github.com/PatrikBorzik2426/BorzikPieces', 'repository_version': '0.1.44'},
        piece_input_kwargs={'subjects': {'type': 'fromUpstream', 'upstream_task_id': 'DataSplitP_54acc3ceb29a423892a76ed1ae55c73d', 'output_arg': 'test_subjects'}, 'output_dir': '/home/shared_storage/medical_data/preprocessed', 'normalization': 'zscore', 'lower_percentile': 1, 'upper_percentile': 99, 'save_as_numpy': True}
    )()
    NiftiPrepr_e1f49198aa3c4456884ce56b8f551911 = Task(
        dag,
        task_id='NiftiPrepr_e1f49198aa3c4456884ce56b8f551911',
        workspace_id=2,
        workflow_shared_storage={'source': 'Local', 'mode': 'Read/Write', 'provider_options': {}},
        container_resources={'requests': {'cpu': '100.0m', 'memory': '128.0Mi'}, 'limits': {'cpu': '100.0m', 'memory': '128.0Mi'}, 'use_gpu': False},
        piece={'name': 'NiftiPreprocessingPiece', 'source_image': 'ghcr.io/borzikpieces/borzikpieces:0.1.44-group0', 'repository_url': 'https://github.com/PatrikBorzik2426/BorzikPieces', 'repository_version': '0.1.44'},
        piece_input_kwargs={'subjects': {'type': 'fromUpstream', 'upstream_task_id': 'DataSplitP_54acc3ceb29a423892a76ed1ae55c73d', 'output_arg': 'train_subjects'}, 'output_dir': '/home/shared_storage/medical_data/preprocessed', 'normalization': 'zscore', 'lower_percentile': 1, 'upper_percentile': 99, 'save_as_numpy': True}
    )()
    PituitaryD_4ac318bb02c14e3fb1641918bde712d8 = Task(
        dag,
        task_id='PituitaryD_4ac318bb02c14e3fb1641918bde712d8',
        workspace_id=2,
        workflow_shared_storage={'source': 'Local', 'mode': 'Read/Write', 'provider_options': {}},
        container_resources={'requests': {'cpu': '100.0m', 'memory': '128.0Mi'}, 'limits': {'cpu': '100.0m', 'memory': '128.0Mi'}, 'use_gpu': False},
        piece={'name': 'PituitaryDatasetPiece', 'source_image': 'ghcr.io/borzikpieces/borzikpieces:0.1.44-group0', 'repository_url': 'https://github.com/PatrikBorzik2426/BorzikPieces', 'repository_version': '0.1.44'},
        piece_input_kwargs={'train_subjects': {'type': 'fromUpstream', 'upstream_task_id': 'NiftiPrepr_e1f49198aa3c4456884ce56b8f551911', 'output_arg': 'preprocessed_subjects'}, 'val_subjects': {'type': 'fromUpstream', 'upstream_task_id': 'NiftiPrepr_5a6c68d92d3642c7823630abcd6f20e6', 'output_arg': 'preprocessed_subjects'}, 'test_subjects': {'type': 'fromUpstream', 'upstream_task_id': 'NiftiPrepr_0e19aa0d1c584cfab52a7428f142b25e', 'output_arg': 'preprocessed_subjects'}, 'data_dir': '/home/shared_storage/medical_data/preprocessed', 'batch_size': 2, 'shuffle_train': True}
    )()

    NiftiVisua_f6bab4518ef3452eb58ef350fbda4ed9.set_upstream([globals()[t] for t in ['NiftiDataL_37b84931888e454aac0ad9800b9a0106']])
    DataSplitP_54acc3ceb29a423892a76ed1ae55c73d.set_upstream([globals()[t] for t in ['NiftiDataL_37b84931888e454aac0ad9800b9a0106']])
    NiftiPrepr_5a6c68d92d3642c7823630abcd6f20e6.set_upstream([globals()[t] for t in ['DataSplitP_54acc3ceb29a423892a76ed1ae55c73d']])
    NiftiPrepr_0e19aa0d1c584cfab52a7428f142b25e.set_upstream([globals()[t] for t in ['DataSplitP_54acc3ceb29a423892a76ed1ae55c73d']])
    NiftiPrepr_e1f49198aa3c4456884ce56b8f551911.set_upstream([globals()[t] for t in ['DataSplitP_54acc3ceb29a423892a76ed1ae55c73d']])
    PituitaryD_4ac318bb02c14e3fb1641918bde712d8.set_upstream([globals()[t] for t in ['NiftiPrepr_e1f49198aa3c4456884ce56b8f551911', 'NiftiPrepr_5a6c68d92d3642c7823630abcd6f20e6', 'NiftiPrepr_0e19aa0d1c584cfab52a7428f142b25e']])
