import numpy as np
import jax
from jax import numpy as jnp
from jax import jit
from jax import lax

# Дадлага : Jax-аар хялбар неорон сүлжээ барих
# https://flax.readthedocs.io/en/latest/notebooks/flax_guided_tour.html

# Неорон сүлжээнд суралцах жингүүдийг оролт гаралтын дүрсээс
# хамааруулан тохируулж үүсгэх
def neural_network_init(
        rng,
        input_features,
        output_features,
        kernel_initialization=jax.nn.initializers.lecun_normal(),
        bias_initialization=jax.nn.initializers.zeros
        ):
    k1, k2 = jax.random.split(rng)
    # Pseudo Random Number Generator Key ашиглан weight-үүд үүсгэх
    kernel = kernel_initialization(k1, (input_features, output_features))
    bias   = bias_initialization(k2, (output_features,))
    return kernel, bias

# Неорон сүлжээг үүсгэсэн жингүүдийн тусламжтайгаар ажиллуулах
def neural_network_apply(parameters, inputs):
    kernel, bias = parameters
    # Y = WX+b, буюу неорон сүлжээний хамгийн энгийн хэлбэр
    return jnp.dot(inputs, kernel) + bias


# JAX нь тодорхой хурдасгуур бүхий төхөөрөмжүүдэд зориулж оновчилсон кодууд
# буюу XLA(Accelerated Linear Algebra) кодруу Just In Time хөрвүүлэгч ашиглан
# хөрвүүлэг хийдэг ийм учраас хөрвүүлэх python код нь pure функц буюу дотроо
# глобал хувьсагч ашигладаггүй функц байх ёстой. Ийм учраас random утга 
# үүсгэгчийг үргэлж гаднаас дамжуулан хэрэглэдэг.


# Неорон сүлжээний жингүүдээ үүсгэж авах
params = neural_network_init(
        jax.random.PRNGKey(0),
        input_features=4,
        output_features=2
        )
print("Неорон сүлжээний жингүүд эхний байдлаар :")
print(params)

# Неорон сүлжээний оролтын өгөгдлийг бэлтгэх
# энд 1 ширхэг дата буюу batch нь 1 гэсэн үг 4-н оролтын өгөгдлийн дүрс
x = jnp.ones((1, 4))
y = jnp.ones((1, 2))
print("Оролтын өгөгдөл :")
print(x)
print("Оролтын өгөгдлийн дүрс :")
print(x.shape)

# Неорон сүлжээгээрээ оролтын өгөгдлөө оруулан forward дуудалт хийх
y = neural_network_apply(params, x)
print("Неорон сүлжээний predict хийсэн гаралт :")
print(y)
print("Гаралтын дүрс :")
print(y.shape)


# Loss функцыг тодорхойлох, квадратуудын дундаж хэрэглэж байгаа учраас L2 loss гэнэ
def loss_fn(parameters, x, y):
    y_predicted = neural_network_apply(parameters, x)
    return jnp.mean(jnp.square(y-y_predicted))

# Параметрүүдээс хамаарсан loss функцын уламжлал буюу дифферинциалчилсан функц
# үүнийг градиент утгууд олж loss-ийн objective буюу зорилгыг биелүүлэхэд хэрэглэнэ
# зорилго гэдэг loss функцыг максимумчилах уу эсвэл минимумчилах уу гэх мэт
# ∇_parameters(Loss) = d(Loss)/d(parameters)
gradient_fn = jax.grad(loss_fn) # JAX хамгийн эхний параметрээс хамааруулан дифференциалчилна

# Градиент утгууд олж харая, параметрүүдийг жаахан өөрчлөхөд loss нэг бол өсч нэг 
# бол буурна, хэрэв зорилго минимумчилах буюу үнэн шошгоруу дөхүүлж неорон сүлжээг
# сургах бол минимумчилах чиглэлрүү заасан градиент утгуудыг ашиглан параметрүүдийг
# шинэчлэх хэрэгтэй байдаг
gradients = gradient_fn(params, x, y)
print("Неорон сүлжээний параметрүүдийг шинэчлэх градиент утгууд :")
print(gradients)


# Санамсаргүй датасет үүсгэн энэ неорон сүлжээг сургах гээд үзье
dataset_size = 10
input_ds     = jax.random.normal(jax.random.PRNGKey(0), shape=(dataset_size, 4))
label_ds     = jax.random.normal(jax.random.PRNGKey(0), shape=(dataset_size, 2))
print("Датасетын оролт :")
print(input_ds)
print("Датасетын гаралт :")
print(label_ds)


# Эдгээр датасетыг ашиглан неорон сүлжээг сургая
def train_neural_network_step(parameters, inputs, labels):
    learning_rate = 0.001
    gradients = jax.grad(jax.jit(loss_fn))(parameters, inputs, labels)
    #print("Сургаж байхад үүссэн градиентүүд :")
    #print(gradients)
    #print("Сургах ёстой параметрүүд :")
    #print(parameters)
    updated_parameters = []
    for param, grad in zip(parameters, gradients):
        updated_parameters.append(param - learning_rate*grad)
    return tuple(updated_parameters)

params = train_neural_network_step(params, input_ds, label_ds)
print("Нэг алхам сургаж шинэчилсэн параметрүүд :")
print(params)


total_steps = 20000 # нийт сургах алхамын тоо
print("Неорон сүлжээг сургаж байна түр хүлээгээрэй...")
for i in range(total_steps):
    params = train_neural_network_step(params, input_ds, label_ds)
print("Сургаж дууслаа.")

# Неорос сүлжээг юм сурсан эсэхийг шалгая
print("--------- MOMENT OF THE TRUTH ---------")
print("Тааварлах ёстой үнэн утгууд:")
print(label_ds)
predicted = neural_network_apply(params, input_ds)
print("Сургасан неорон сүлжээний гаралтын утгууд :")
print(predicted)
print("--------------- THE END ---------------")
