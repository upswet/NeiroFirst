package neiro2_layer_event;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Описывает одиночный нейрон
 */
public class Neiro implements Serializable {
    TypeEnum typeEnum; //тип нейрона
    FunEnum fun = FunEnum.NONE; //тип функции активации
    List<Link> inputs = new ArrayList<>(); //входные связи
    int calcInputRecive = 0; //кол-во входящих связей по которым получили сигнал при прямом проходе
    List<Link> outputs = new ArrayList<>(); //выходные связи
    int calcOutputDeltaCalc = 0;//количество выходных связей для которых связанный нейрон уже вычислил дельту-ошибки в процессе обратного распространения ошибки

    StatusNeironEnum status = StatusNeironEnum.READY;
    Layer layer;

    float valueInput; //входящий сигнал - сумма сигналов полученным по всем входящим связям
    float valueOutput; //исходящий сигнал - результат применения функции активации к входящему сигналу
    float target; //значение правильного ответа для выходного нейрона для обчения
    float delta; //дельта. Участвует в корректировке весов в алгоритме обратного распространения ошибки


    public Neiro(TypeEnum typeEnum, FunEnum fun) {
        this.typeEnum = typeEnum;
        this.fun = fun;

        if (typeEnum.equals(TypeEnum.BIAS))
            valueInput = 1F;
        else
            valueInput = 0F;
    }

    public Neiro(TypeEnum typeEnum, FunEnum fun, Layer layer) {
        this(typeEnum, fun);
        if (typeEnum.equals(TypeEnum.INPUT) || typeEnum.equals(TypeEnum.OUTPUT))
            this.layer = layer;
    }

    public void setTarget(float target) {
        if (!typeEnum.equals(TypeEnum.OUTPUT))
            throw new RuntimeException("Нельзя помещать целевое значение в любой нейрон кроме выходного");
        this.target = target;
    }

    /**
     * Поместить значение в нейрон. Только для входных нейронов
     */
    public void setValue(float valueInput) {
        if (!typeEnum.equals(TypeEnum.INPUT))
            throw new RuntimeException("Нельзя помещать входные данные в любой нейрон кроме входного");
        this.valueInput = valueInput;
    }

    /**
     * Получить выходное значение нейрона
     */
    public float getValue() {
        return this.valueOutput;
    }

    /**
     * Цикл прямого распространения сигнала
     *
     * @return - истину если цикл закончен для данного нейрона и ложь в противном случае
     */
    public boolean forwardProcess() {
        switch (status) {
            case READY, MY_DELTA_CALC -> {
                if (this.typeEnum.equals(TypeEnum.INPUT) || this.typeEnum.equals(TypeEnum.BIAS)) {
                    this.valueOutput = this.fun(valueInput);
                    this.outputs.forEach(link -> link.sendSignal(valueOutput));
                    this.status = StatusNeironEnum.SEND_OUTPUT_ALL;
                    return true;
                } else if (calcInputRecive > 0) {
                    this.status = StatusNeironEnum.RECIVE_INPUT;
                    return forwardProcess();
                }
            }
            case RECIVE_INPUT -> {
                if (calcInputRecive >= this.inputs.size()) {
                    this.status = StatusNeironEnum.RECIVED_INPUT_ALL;
                    calcInputRecive = 0;
                    return forwardProcess();
                }
            }
            case RECIVED_INPUT_ALL -> {
                valueOutput = fun(valueInput);
                valueInput = 0F;//т.к. суммирование входов уже зкончилось ибо пришло время вычислять выход

                //передадим сигнал по всем исходящим связям
                for (Link link : outputs)
                    link.sendSignal(this.getValue());

                if (this.typeEnum.equals(TypeEnum.OUTPUT))
                    this.layer.neiroOnFrward();

                this.status = StatusNeironEnum.SEND_OUTPUT_ALL;
                return true; //обработка прямого прохода для данного нейрона закончилась
            }

            case SEND_OUTPUT_ALL -> {
                return true;
            }
        }

        return false;
    }

    /**
     * Цикл обратного распространения ошибки и коррекции весов исходящих связей нейрона
     *
     * @param lr     - коэффициент обучения
     * @param moment - момент обучения
     * @return - истину если цикл закончен для данного нейрона и ложь в противном случае
     */
    public boolean backwardProcess(float lr, float moment) {
        switch (status) {
            case SEND_OUTPUT_ALL -> {
                if (this.typeEnum.equals(TypeEnum.OUTPUT)) {
                    float err = target - getValue();
                    delta = err * derivative(valueOutput);

                    this.status = StatusNeironEnum.MY_DELTA_CALC;
                    this.inputs.forEach(Link::checkDelta); //скажем всем входным связям, что мы вычислили свою дельту ошибки
                    return true;
                } else if (calcOutputDeltaCalc > 0) {
                    this.status = StatusNeironEnum.OUTPUT_DELTA_CALC;
                    return backwardProcess(lr, moment);
                }
            }
            case OUTPUT_DELTA_CALC -> {
                if (calcOutputDeltaCalc >= this.outputs.size()) {
                    this.status = StatusNeironEnum.OUTPUT_DELTA_CALC_ALL;
                    calcOutputDeltaCalc = 0;
                    return backwardProcess(lr, moment);
                }
            }
            case OUTPUT_DELTA_CALC_ALL -> {
                //вычислим свою дельту
                if (this.typeEnum.equals(TypeEnum.HIDDEN)) {
                    delta = 0F;
                    for (Link link : outputs) {
                        Neiro neiroOutput = link.output;
                        delta += neiroOutput.delta * link.weight;

                        link.weightCorrect(lr, moment);
                    }
                    delta = derivative(valueOutput) * delta;
                    this.inputs.forEach(Link::checkDelta); //скажем всем входным связям, что мы вычислили свою дельту ошибки
                }

                if (this.typeEnum.equals(TypeEnum.INPUT) || this.typeEnum.equals(TypeEnum.BIAS))
                    for (Link link : outputs)
                        link.weightCorrect(lr, moment);

                this.status = StatusNeironEnum.MY_DELTA_CALC;
                if (this.typeEnum.equals(TypeEnum.INPUT))
                    layer.neiroOnBackward();
                return true;
            }
            case MY_DELTA_CALC -> {
                return true;
            }
        }
        return false;
    }

    /**
     * Событие получения нейроном сигнала по одной из входящих связей при прямом проходе
     */
    public void onRecive(float value) {
        valueInput += value;
        calcInputRecive++;
    }

    /**
     * Событие олучения нейроном сигнала по одной из исходящих связей, что нейрон на другом её конце вычислил свою дельту-ошибки при обратном проходе
     */
    public void onOutputDelteCalc() {
        calcOutputDeltaCalc++;
    }

    /**
     * функция активации
     */
    public float fun(float v) {
        switch (fun) {
            case SIGMOID -> {
                return (float) (1.0 / (1 + Math.exp(-v)));
            }
            default -> {
                return v;
            }
        }
    }

    /**
     * Производная функции активации
     */
    public float derivative(float v) {
        switch (fun) {
            case SIGMOID -> {
                return v * (1 - v);
            }
            default -> {
                return v;
            }
        }
    }
}
