package neiro3_virual_process;

/**Статус нейросети*/
public enum StatusNetEnum {
    READY, //базовый статус
    FORWARD_RUN, //выполняется прямой проход
    FORWARD_END, //прямой проход завершён. Можно брать данные с выходных нейронов
    BACKWARD_RUN,//выполняется обратный проход
}
