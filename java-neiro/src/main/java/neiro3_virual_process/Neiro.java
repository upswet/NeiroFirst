package neiro3_virual_process;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Описывает одиночный нейрон
 */
public class Neiro implements Serializable {
    TypeEnum typeEnum; //тип нейрона
    FunEnum fun = FunEnum.NONE; //тип функции активации
    List<Link> inputs = new ArrayList<>(); //входные связи
    AtomicInteger calcInputRecive = new AtomicInteger(0); //кол-во входящих связей по которым получили сигнал при прямом проходе
    List<Link> outputs = new ArrayList<>(); //выходные связи
    AtomicInteger calcOutputDeltaCalc = new AtomicInteger(0);//количество выходных связей для которых связанный нейрон уже вычислил дельту-ошибки в процессе обратного распространения ошибки

    StatusNeironEnum status = StatusNeironEnum.READY;
    Layer layer;

    volatile float valueInput; //входящий сигнал - сумма сигналов полученным по всем входящим связям
    volatile float valueOutput; //исходящий сигнал - результат применения функции активации к входящему сигналу
    volatile float target; //значение правильного ответа для выходного нейрона для обчения
    volatile float delta; //дельта. Участвует в корректировке весов в алгоритме обратного распространения ошибки


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


    private void sendSignal() {
        this.valueOutput = this.fun(valueInput);
        this.outputs.forEach(link -> link.sendSignal(valueOutput));
        calcInputRecive.set(0); //т.к. суммирование входов уже зкончилось
        if (!this.typeEnum.equals(TypeEnum.BIAS))
            valueInput = 0F;//т.к. суммирование входов уже зкончилось ибо пришло время вычислять выход
    }

    /**
     * Цикл жизни нейрона. Соединённые вместе прямое и обратное распространение ошибки
     *
     * @param lr     - коэффициент обучения
     * @param moment - момент обучения
     */
    public void process(float lr, float moment) {
        switch (typeEnum) {
            case INPUT, BIAS -> {
                switch (status) {
                    //forward
                    case READY -> {
                        this.sendSignal();
                        this.status = StatusNeironEnum.SEND_OUTPUT_ALL;
                    }
                    //backward
                    case SEND_OUTPUT_ALL -> {
                        if (calcOutputDeltaCalc.get() > 0) {
                            this.status = StatusNeironEnum.OUTPUT_DELTA_CALC;
                            process(lr, moment);
                        }
                    }
                    case OUTPUT_DELTA_CALC -> {
                        if (calcOutputDeltaCalc.get() >= this.outputs.size()) {
                            this.status = StatusNeironEnum.OUTPUT_DELTA_CALC_ALL;
                            calcOutputDeltaCalc.set(0);
                            process(lr, moment);
                        }
                    }
                    case OUTPUT_DELTA_CALC_ALL -> {
                        for (Link link : outputs)
                            link.weightCorrect(lr, moment);

                        this.status = StatusNeironEnum.MY_DELTA_CALC;
                        if (this.typeEnum.equals(TypeEnum.INPUT))
                            layer.neiroOnBackward();
                    }
                }
            }
            case HIDDEN -> {
                switch (status) {
                    //forward
                    case READY, MY_DELTA_CALC -> {
                        if (calcInputRecive.get() > 0) {
                            this.status = StatusNeironEnum.RECIVE_INPUT;
                            process(lr, moment);
                        }
                    }
                    case RECIVE_INPUT -> {
                        if (calcInputRecive.get() >= this.inputs.size()) {
                            this.status = StatusNeironEnum.RECIVED_INPUT_ALL;
                            calcInputRecive.set(0);
                            process(lr, moment);
                        }
                    }
                    case RECIVED_INPUT_ALL -> {
                        this.sendSignal();

                        this.status = StatusNeironEnum.SEND_OUTPUT_ALL;
                    }

                    //backward
                    case SEND_OUTPUT_ALL -> {
                        if (calcOutputDeltaCalc.get() > 0) {
                            this.status = StatusNeironEnum.OUTPUT_DELTA_CALC;
                            process(lr, moment);
                        }
                    }
                    case OUTPUT_DELTA_CALC -> {
                        if (calcOutputDeltaCalc.get() >= this.outputs.size()) {
                            this.status = StatusNeironEnum.OUTPUT_DELTA_CALC_ALL;
                            calcOutputDeltaCalc.set(0);
                            process(lr, moment);
                        }
                    }
                    case OUTPUT_DELTA_CALC_ALL -> {
                        //вычислим свою дельту
                        delta = 0F;
                        for (Link link : outputs) {
                            Neiro neiroOutput = link.output;
                            delta += neiroOutput.delta * link.weight;

                            link.weightCorrect(lr, moment);
                        }
                        delta = derivative(valueOutput) * delta;
                        this.inputs.forEach(Link::checkDelta); //скажем всем входным связям, что мы вычислили свою дельту ошибки

                        this.status = StatusNeironEnum.MY_DELTA_CALC;
                    }
                }
            }
            case OUTPUT -> {
                switch (status) {
                    //forward
                    case READY, MY_DELTA_CALC -> {
                        if (calcInputRecive.get() > 0) {
                            this.status = StatusNeironEnum.RECIVE_INPUT;
                            process(lr, moment);
                        }
                    }
                    case RECIVE_INPUT -> {
                        if (calcInputRecive.get() >= this.inputs.size()) {
                            this.status = StatusNeironEnum.RECIVED_INPUT_ALL;
                            calcInputRecive.set(0);
                            process(lr, moment);
                        }
                    }
                    case RECIVED_INPUT_ALL -> {
                        this.sendSignal();
                        this.layer.neiroOnFrward();
                        this.status = StatusNeironEnum.SEND_OUTPUT_ALL;
                    }

                    //backward
                    case SEND_OUTPUT_ALL -> {
                        if (target != 666) {
                            float err = target - getValue();
                            delta = err * derivative(valueOutput);
                            target = 666; //будем исп это значение как флаг того, что нам не нужен режим обратного распостранения ибо это у нас не тренировка, а обычная работа уже натр сети

                            this.status = StatusNeironEnum.MY_DELTA_CALC;
                            this.inputs.forEach(Link::checkDelta); //скажем всем входным связям, что мы вычислили свою дельту ошибки
                        }
                    }
                }
            }
        }
    }


    /**
     * Событие получения нейроном сигнала по одной из входящих связей при прямом проходе
     */
    public void onRecive(float value) {
        valueInput += value;
        calcInputRecive.incrementAndGet();
    }

    /**
     * Событие олучения нейроном сигнала по одной из исходящих связей, что нейрон на другом её конце вычислил свою дельту-ошибки при обратном проходе
     */
    public void onOutputDelteCalc() {
        calcOutputDeltaCalc.incrementAndGet();
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
