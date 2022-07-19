
import os
import kflearn as kfl
import train_lgb

# os.environ['kfl_activate_component'] = 'train_node'
if __name__ == '__main__':
    # 获取上下文
    context = kfl.get_context_v2()

    # 获取本节点输入及配置
    train_path = context.get_input('train_data')
    test_path = context.get_input('test_data')
    run_dir = context['run_dir']
    dt = context['dt']
    lr = context.get_config('lr')

    # 执行训练
    model_path = os.path.join(run_dir, 'model.bst')
    os.makedirs('model', exist_ok=True)
    train_lgb.train(train_path, test_path, model_path)

    # 设置输出
    context.set_output('model_path', model_path)
    context.set_output('metrics_horn', {'auc': 0.88, 'recall': 0.1, 'f1': 0.8})

    # 序列化（必填）
    # context.dump()
