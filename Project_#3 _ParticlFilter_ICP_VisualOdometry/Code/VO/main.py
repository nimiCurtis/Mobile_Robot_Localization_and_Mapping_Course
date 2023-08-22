from project_questions import ProjectQuestions


if __name__ == "__main__":
    vo_data = {}
    vo_data['dir'] = r"data"
    ## insert results dir! 
    vo_data['results'] = r"..\Results\VO"
    vo_data['sequence'] = 2
    
    project = ProjectQuestions(vo_data)
    project.run()