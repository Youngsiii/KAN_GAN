import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,  # 输入特征
        out_features,   # 输出特征
        grid_size=5,    # 网格个数/控制点个数
        spline_order=3,   # B样条阶数
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,      # base激活函数是SiLU，论文上面有
        grid_eps=0.02,
        grid_range=[-1, 1],    # 网格范围
    ):
        super(KANLinear, self).__init__()
        # 前几个参数直接赋值
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # h = (1 + 1)/5 = 0.4 每个网格的大小
        h = (grid_range[1] - grid_range[0]) / grid_size
        # 这里的grid不太理解 可能上面的grid_size为控制点个数，这里的grid为节点向量
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h    # torch.arange()只有start和end步长默认step=1
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()   # 改变了多元数组在内存中的存储顺序，以便配合view方法使用
        )
        self.register_buffer("grid", grid)   # 定义一组参数，该参数在模型训练时不会更新，但在保存模型时，该参数又作为模型参数不可或缺的一部分被保存
        # torch.nn.Parameter(tensor, requires_grad=True)  将不可训练的tensor变成可训练的parameter,告诉pytorch这个tensor要被优化，参与梯度计算和反向传播
        # 存储base函数参数
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        # 存储spline函数参数
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        # 样条函数缩放系数，对应论文中的w，如果需要，就创造出来并将其变成可训练的parameter
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        # 只要是定义的新的模块，都要在__init__函数中使用self.reset_parameters()函数用来将模型参数初始化到合适的状态便于后续训练，一般常用kaiming初始化或者Xavier初始化
        self.reset_parameters()


    # 定义了KANLinear类中的reset_parameters方法，用于自定义模型权重参数的初始化方法，它展示了如何针对不同的权重参数应用特定的初始化策略
    def reset_parameters(self):
        # Kaiming均匀初始化，第一项为权重张量，a表示这层之后使用的整流器的斜率(用于计算分布的范围参数)，第三项mode默认为“fan_in”即保持前向传播时的权重方差大小，如果选用"fan_out"则维持后向传播时的权重方差大小，第四项非线性激活函数默认为"LeakyReLU"
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():  # 上下文管理器，禁用自动梯度计算，防止这些操作被记录到计算图
            noise = (    # 生成噪声
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)  # torch.rand生成[0,1]之间均匀分布的随机数
                    - 1 / 2     # 减去1/2后生成[-0.5,0.5]之间的随机数
                )
                * self.scale_noise    # 乘以噪声缩放系数进行缩放
                / self.grid_size      # 除以grid_size进行归一化
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)  # 考虑对spline_scaler进行常数初始化
                # 还是采用Kaiming均匀分布初始化来初始化spline_scalar
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.   根据给定的输入张量，计算B样条基函数

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).  x是输入张量，形状为(batch_size, in_features)

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).  返回B样条基函数张量，形状为(batch_size, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)  # grid_size+2*spline_order+1可能反映节点的个数
        # 扩展x的维度，通过unsqueeze方法在x的最后一个维度上添加一个新的维度，使x的维度从(batch_size,in_features)变成(batch_size,in_features,1)
        x = x.unsqueeze(-1)

        # 计算初始的B样条基函数   (x >= grid[:, :-1]) & (x < grid[:, 1:])是布尔变量，表示x落在每个样条区间内的情况，然后将其转变为与x一样的数据类型
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        # 递归计算更高阶的B样条基函数，从1次基函数计算到spline_order次
        for k in range(1, self.spline_order + 1):   # 基函数阶数等于次数+1
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )    # 该递推公式基于De Boor Cox算法

        assert bases.size() == (
            x.size(0),     # x.size(0)为batch_size
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()    # 将bases在内存中连续化

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):  # 给定输入点x和输出点y计算插值曲线的系数
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).  输入张量，形状为(batch_size, in_features)
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).  输出张量，形状为(batch_size, in_features, out_features)

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        # 参数检查，确保x和y符合要求
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        # 计算B样条基函数矩阵
        A = self.b_splines(x).transpose(
            0, 1     # 将维度调换一下
        )  # (in_features, batch_size, grid_size + spline_order)

        # B为输出张量y将维度调换一下，与B样条基函数矩阵A保持一致
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)


        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property          # @property装饰器可以让下面定义的方法像属性一样直接获取函数的值
    def scaled_spline_weight(self):    # 对样条函数权重参数进行缩放，如果self.enable_standalone_scale_spline为True就乘以self.spline_scaler，否则乘以1
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features    # 输入x的形状为(batch_size,in_features)
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)   # 先通过激活函数SiLU，再进行base_weight线性变换，而且线性变换的权重类似MLP可学习
        spline_output = F.linear(                                  # 先通过样条函数进行激活，再通过spline_weight线性变换，线性变换的权重spline_weight可学习
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output      # 输出是基函数SiLU输出值和样条函数值之和
        
        output = output.view(*original_shape[:-1], self.out_features)    # 将输出的形状改成想要的形状
        return output

    @torch.no_grad()       # @torch.no_grad()装饰器用于指示pytorch在执行被装饰的函数时不应跟踪函数内部的操作以便进行梯度计算
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )



'''
这段代码是使用PyTorch框架定义模型权重参数的一部分，涉及到深度学习中的神经网络权重初始化。具体来说，它定义了两种类型的权重参数，这些参数通常是用来在神经网络中执行线性变换或者更复杂的运算。下面对每一部分进行详细解释：

### 1. `self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))`

- **`torch.nn.Parameter`**: 这是一个特殊的Tensor，它告诉PyTorch这个张量是要被优化的，即在网络训练过程中，它的值会通过反向传播和优化器更新。这意味着它会参与到模型的梯度计算和权重更新中。

- **`torch.Tensor(out_features, in_features)`**: 这里创建了一个形状为`(out_features, in_features)`的张量，用于存储权重值。这个张量的尺寸暗示着它被用在一个全连接层（Dense Layer）或者线性变换中，其中：
  - `in_features`代表输入特征的数量，即前一层神经元的数量。
  - `out_features`代表输出特征的数量，即这一层神经元的数量。

这个`base_weight`很可能是用于一个标准的线性变换，比如从一个特征空间映射到另一个特征空间。

### 2. `self.spline_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))`

这部分代码定义了另一个参数`self.spline_weight`，其结构相比`base_weight`更为复杂，表明它可能用于实现一些非线性的转换或适应性特征学习。

- **三维张量**: 与`base_weight`不同，这里创建的是一个三维张量，其形状为`(out_features, in_features, grid_size + spline_order)`。这暗示了权重不仅依赖于输入和输出的特征维度，还依赖于额外的两个维度：`grid_size`和`spline_order`。
  
  - `grid_size`: 这个参数通常与某种网格或离散化空间相关，暗示着可能使用了一种基于网格的函数近似或插值技术，如样条插值。
  - `spline_order`: 表示样条函数的阶数。样条插值是一种平滑曲线拟合技术，阶数决定了样条曲线的复杂度和平滑度。常见的样条有线性样条、二次样条、三次样条等，阶数越高，曲线越能逼近复杂的形状。

综合来看，`self.spline_weight`可能用于实现一种自适应或非线性的权重调整机制，这种机制可能与样条函数有关，用于增强模型的表达能力，尤其是在处理非线性关系或需要在特定维度上进行灵活建模的场景。例如，在一些高级的特征转换层中，可能会用到这样的权重来实现更复杂的特征交互或适应性滤波效果。


'''



