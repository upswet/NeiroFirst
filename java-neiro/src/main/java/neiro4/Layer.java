package neiro4;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.atomic.AtomicInteger;

/**Описывает слой как фабрику нейронов*/
public class Layer implements Serializable {

    List<Neiro> neiros=new LinkedList<>(); //упорядоченная совокупность нейронов

    public Layer() { }

    /**
     * Создание входного слоя
     * @param neiroCount - число нейроно в создаваемом слое
     * @param fun - функция активации для них
     * @param isBias - если истина, то добавляем нейрон смещения в слой
     * @return - созданный слой
     */
    public static Layer createLayerInput(Integer neiroCount, FunEnum fun, boolean isBias) {
        Layer layer = new Layer();
        //создание обычных нейронов
        for (int i = 0; i < neiroCount; i++)
            layer.neiros.add(new Neiro(TypeEnum.INPUT, fun));

        if (isBias)
            layer.neiros.add(new Neiro(TypeEnum.BIAS, FunEnum.NONE));

        return layer;
    }

    /**
     * Создание полносвязанного (т.е. каждый нейрон создаваемого слоя будет связан с каждым нейроном предыдущего слоя) скрытого слоя
     * @param previousLayer - предыдущий слой
     * @param neiroCount - число нейроно в создаваемом слое
     * @param fun - функция активации для них
     * @param isBias - если истина, то добавляем нейрон смещения в слой
     * @return - созданный слой
     */
    public static Layer createLayerHidden(Integer neiroCount, FunEnum fun, Layer previousLayer, boolean isBias) {
        Layer layer = new Layer();
        //создание обычных нейронов
        for (int i = 0; i < neiroCount; i++) {
            Neiro neiro = new Neiro(TypeEnum.HIDDEN, fun);
            layer.neiros.add(neiro);
            //создание связей каждого нейрона предыдщего слоя с только что созданным
            for (Neiro neiroForPreviousLayer : previousLayer.neiros)
                if (!neiroForPreviousLayer.typeEnum.equals(TypeEnum.BIAS)) {
                    Link link = new Link(neiroForPreviousLayer, neiro, Link.generateWeight());
                }
        }

        if (isBias)
            layer.neiros.add(new Neiro(TypeEnum.BIAS, FunEnum.NONE));

        return layer;
    }

    /**
     * Создание полносвязанного (т.е. каждый нейрон создаваемого слоя будет связан с каждым нейроном предыдущего слоя) выходного слоя
     * @param previousLayer - предыдущий слой
     * @param neiroCount - число нейроно в создаваемом слое
     * @param fun - функция активации для них
     * @return - созданный слой
     */
    public static Layer createLayerOutput(Integer neiroCount, FunEnum fun, Layer previousLayer) {
        Layer layer = new Layer();
        //создание обычных нейронов
        for (int i = 0; i < neiroCount; i++) {
            Neiro neiro = new Neiro(TypeEnum.OUTPUT, fun);
            layer.neiros.add(neiro);
            //создание связей каждого нейрона предыдщего слоя с только что созданным
            for (Neiro neiroForPreviousLayer : previousLayer.neiros)
                if (!neiroForPreviousLayer.typeEnum.equals(TypeEnum.BIAS)) {
                    Link link = new Link(neiroForPreviousLayer, neiro, Link.generateWeight());
                }
        }

        return layer;
    }
}
