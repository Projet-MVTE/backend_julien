import json
import matplotlib.pyplot as plt

class Node():
    def __init__(self, name, coord=""):
        self.name = name
        self.children = []
        self.coord = coord

    def add_children(self, child):
        if isinstance(child, Node):
            self.children.append(child)
        else:
            print("Erreur : Il faut un Node en parametre")

    def supp_child(self, child):
        if isinstance(child, Node):
            if child in self.children:
                self.children.pop(self.children.index(child))
            else:
                print("Noeud inexistant")
        else:
            print("Erreur : Il faut un Node en parametre")

    def __str__(self):
        return "Noeud {"+self.name+"}"


class Tree():
    separateur = "_;_" #dans le fichier
    def __init__(self):
        self.starting_node = None
        self.file_name = None

    def add_node(self, name, parent):
        """
        Parametres :
        ---------------------------------------
        name   : str  : Nom du noeud à rajouter
        parent : Node : Noeud parrent
        """
        n = Node(name, parent.coord + str(len(parent.children)))
        parent.add_children(n)

    def supp_node(self, node_to_supp):
        """
        Parametres :
        ----------------------------------------
        node_to_supp : Node :  Noeud à supprimer
        """
        if isinstance(node_to_supp, Node):
            pile = [self.starting_node]
            while len(pile)>0:
                node = pile.pop(0)

                if node_to_supp in node.children:
                    node.supp_child(node_to_supp)
                    break
                else:
                    for c in node.children:
                        pile.append(c)
        else:
            print("Erreur : Il faut un Node en parametre")
        self.actualiser_coord()

    def actualiser_coord(self):
        """
        Actualise les coordonees des noeuds
        """
        pile = [self.starting_node]
        while len(pile)>0:
            node = pile.pop(0)
            for k in range(len(node.children)):
                pile.append(node.children[k])
                node.children[k].coord = node.children[k].coord[:-1] + str(k)
            
    def read_file(self, file_name):
        self.file_name = file_name
        f = open(file_name, "r")
        contenu = {l.strip().split(Tree.separateur)[0]:l.strip().split(Tree.separateur)[1] for l in f.readlines()}
        f.close()

        if "0" in contenu:
            self.starting_node = Node(contenu["0"], "0")
            contenu["0"] = self.starting_node
            for key in contenu:
                if key !="0":
                    contenu[key] = Node(contenu[key], key)
                    contenu[key[:-1]].add_children(contenu[key])
        else:
            print("ERREUR : Aucun noeud de depart dans le fichier")
       
    def show(self):
        if self.starting_node != None:
            pile = [self.starting_node] #initialisation de la pile
            while len(pile)>0:
                node = pile.pop(0)
                
                for c in node.children:
                    pile.append(c)
                    plt.plot([len(node.coord), len(c.coord)],[int(node.coord[-1]),int(c.coord[-1])])

                plt.text(len(node.coord),int(node.coord[-1]), node.name)

            plt.xlim(-1,10)
            plt.ylim(-1,10)
            plt.show()
            
        else:
            print("ERREUR : Arbre vide")

    def save_txt(self, f_name=None):
        chaine = ""
        pile = [self.starting_node]

        while len(pile)>0:
            node = pile.pop(0)
            for c in node.children:
                pile.append(c)
            chaine += node.coord+Tree.separateur+node.name+"\n"
        
        if f_name != None:
            file_name = f_name + ".txt"
        elif self.file_name != None:
            file_name = self.file_name
        else :
            print("ERREUR : Pas de nom de fichier")
            return 0

        file = open(file_name, "w")
        file.write(chaine)
        file.close()

    def save_json_old(self, file_name):
        """
        Enregistre l'arbre au format .json
        """
        def insert_child(dic, parent, child):
            """
            Cherche recursivement le noeud dans lequel ajouter le child
            """
            if parent.name==dic["name"]:
                dic["children"].append({"name":child.name, "children":[]})
            else:
                for k in range(len(dic["children"])):
                    insert_child(dic["children"][k], parent, child)
            
        
        pile = [self.starting_node]
        dic = {"name":self.starting_node.name, "children":[]}

        while len(pile)>0:
            node = pile.pop(0)
            for c in node.children:
                pile.append(c)
                insert_child(dic, node, c)
        
        f = open(file_name+".json", "w")
        f.write(json.dumps(dic, indent=4))
        f.close()

    def save_json(self, file_name):

        

        pile = [self.starting_node]
        dic = {"message":"Buil by Dina, Julien, Johan, Jawad and Naim at ENSTA Bretagne",
               "source":"serveur",
               "lastUpdated":None,
               "data":{}}

        while len(pile)>0:
            node = pile.pop(0)
            dic["data"][node.coord] = {"id":node.coord,
                                       "title":node.name,
                                       "type":1*(len(node.children)==0),
                                       "next":[c.coord for c in node.children]}
            for c in node.children:
                pile.append(c)

        f = open(file_name+".json", "w")
        f.write(json.dumps(dic, indent=4))
        f.close()
        

if __name__=="__main__":
    arbre = Tree()
    arbre.read_file("arbre_decisions.txt")
    arbre.show()
    arbre.add_node("symptome5", arbre.starting_node)
    arbre.show()
    arbre.save_json("test_json")
    
    #arbre.supp_node(arbre.starting_node.children[-2])
    #arbre.show()
    arbre.save_txt("test")
    
