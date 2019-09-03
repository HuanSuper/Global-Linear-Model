# Global-Linear-Model
<ol>
  <li>简介</li>使用Global Linear Model模型预测当前句子的词性序列
  <li>Global Linear Model模型</li>
  <ul>Linear Model与HMM模型的结合，进行词性序列预测时考虑整个句子（前一个词性对后一个词性的影响），确定一个句子的最佳词性序列
    <li>特征提取</li>添加一个新特征：当前词性+前一个词性
    <li>权重学习</li>同Linear Model的在线学习方法
    <li>句子最佳词性序列</li>同HMM模型（维特比算法）
  </ul>
  <li>评价指标</li>准确率 = 正确的标注数 / 总的标注数
  <li>程序</li>
  <ol>
    <li>数据</li>
    训练集：train.conll<br>
    测试集：dev.conll
    <li>代码</li>
    global_linear_model.py:实现Global Linear Model模型词性标注
  </ol>
</ol>
